"""
LILA-style latent action model for EMG teleoperation (structure only).

Components (see proposal):
  - ActionEncoder E_a: (hand_state, hand_velocity) -> 2D latent. No language.
  - EMGNetwork f_emg: EMG window (320) -> 2D latent.
  - ActionDecoder D: (hand_state, latent, language_embedding) -> synergy velocity;
    hidden layers use FiLM conditioning from language.

Training stages are implemented elsewhere; this module only defines ``nn.Module``s.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """Map language embedding to per-feature scale and shift (gamma, beta)."""

    def __init__(self, language_dim: int, feature_dim: int, hidden_dim: int | None = None):
        super().__init__()
        h = hidden_dim if hidden_dim is not None else feature_dim
        self.net = nn.Sequential(
            nn.Linear(language_dim, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, 2 * feature_dim),
        )
        self.feature_dim = feature_dim

    def forward(self, language_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            language_embedding: (batch, language_dim)
        Returns:
            gamma, beta each (batch, feature_dim)
        """
        gb = self.net(language_embedding)
        gamma, beta = gb.split(self.feature_dim, dim=-1)
        return gamma, beta


class ActionEncoder(nn.Module):
    """
    E_a: compress current synergy state + ground-truth synergy velocity to a 2D latent.
    Used only in training (Stage 1 target for EMG; Stage 2 teacher). Does not see language.
    """

    def __init__(
        self,
        synergy_dim: int,
        latent_dim: int = 2,
        hidden_dim: int = 256,
        num_hidden: int = 2,
    ):
        super().__init__()
        self.synergy_dim = synergy_dim
        self.latent_dim = latent_dim
        in_dim = 2 * synergy_dim
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_hidden):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)]
            d = hidden_dim
        layers.append(nn.Linear(d, latent_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, hand_state: torch.Tensor, hand_velocity: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hand_state: (batch, synergy_dim)
            hand_velocity: (batch, synergy_dim)
        Returns:
            latent: (batch, latent_dim)
        """
        x = torch.cat([hand_state, hand_velocity], dim=-1)
        return self.mlp(x)


class EMGNetwork(nn.Module):
    """
    f_emg: EMG window flattened (8 * 40 = 320) -> 2D latent (same space as ActionEncoder).
    """

    def __init__(
        self,
        emg_dim: int = 320,
        latent_dim: int = 2,
        hidden_dim: int = 256,
        num_hidden: int = 2,
    ):
        super().__init__()
        self.emg_dim = emg_dim
        self.latent_dim = latent_dim
        layers: list[nn.Module] = []
        d = emg_dim
        for _ in range(num_hidden):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)]
            d = hidden_dim
        layers.append(nn.Linear(d, latent_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, emg_window: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emg_window: (batch, emg_dim) or (batch, 8, 40) — flattened if 3D
        Returns:
            latent: (batch, latent_dim)
        """
        if emg_window.dim() == 3:
            emg_window = emg_window.reshape(emg_window.shape[0], -1)
        return self.mlp(emg_window)


class ActionDecoder(nn.Module):
    """
    D: (hand_state, latent) -> synergy velocity; language_embedding modulates hidden
    activations via FiLM. Set ``use_language=False`` for a no-FiLM baseline (same MLP width).
    """

    def __init__(
        self,
        synergy_dim: int,
        latent_dim: int = 2,
        language_dim: int = 768,
        hidden_dim: int = 256,
        num_film_layers: int = 2,
        use_language: bool = True,
    ):
        super().__init__()
        self.synergy_dim = synergy_dim
        self.latent_dim = latent_dim
        self.language_dim = language_dim
        self.hidden_dim = hidden_dim
        self.num_film_layers = num_film_layers
        self.use_language = use_language

        self.fc_in = nn.Linear(synergy_dim + latent_dim, hidden_dim)

        if use_language:
            self.films = nn.ModuleList(
                FiLM(language_dim, hidden_dim, hidden_dim=hidden_dim) for _ in range(num_film_layers)
            )
        else:
            self.films = None

        self.fc_hidden = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_film_layers - 1)
        )
        self.fc_out = nn.Linear(hidden_dim, synergy_dim)

    def forward(
        self,
        hand_state: torch.Tensor,
        latent: torch.Tensor,
        language_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hand_state: (batch, synergy_dim)
            latent: (batch, latent_dim)
            language_embedding: (batch, language_dim); required if ``use_language``
        Returns:
            velocity_hat: (batch, synergy_dim)
        """
        if self.use_language:
            if language_embedding is None:
                raise ValueError("language_embedding is required when use_language=True")
        x = torch.cat([hand_state, latent], dim=-1)
        h = self.fc_in(x)

        for i in range(self.num_film_layers):
            if self.use_language and self.films is not None:
                gamma, beta = self.films[i](language_embedding)
                h = gamma * h + beta
            h = F.relu(h, inplace=True)
            if i < self.num_film_layers - 1:
                h = self.fc_hidden[i](h)

        return self.fc_out(h)


class LilaTeleopModel(nn.Module):
    """
    Full graph: ``encoder``, ``emg``, ``decoder``.

    Typical forwards:
      - Stage 1: z = encoder(s, v); v_hat = decoder(s, z, lang)
      - Stage 2: z_star = encoder(s, v); z_emg = emg(emg_window); loss = ||z_emg - z_star||
      - Inference (EMG path): z = emg(emg_window); v_hat = decoder(s, z, lang)
    """

    def __init__(
        self,
        synergy_dim: int,
        latent_dim: int = 2,
        emg_dim: int = 320,
        language_dim: int = 768,
        hidden_dim: int = 256,
        encoder_hidden_layers: int = 2,
        emg_hidden_layers: int = 2,
        decoder_film_layers: int = 2,
        decoder_use_language: bool = True,
    ):
        super().__init__()
        self.synergy_dim = synergy_dim
        self.latent_dim = latent_dim
        self.emg_dim = emg_dim
        self.language_dim = language_dim

        self.encoder = ActionEncoder(
            synergy_dim, latent_dim, hidden_dim=hidden_dim, num_hidden=encoder_hidden_layers
        )
        self.emg = EMGNetwork(
            emg_dim=emg_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, num_hidden=emg_hidden_layers
        )
        self.decoder = ActionDecoder(
            synergy_dim=synergy_dim,
            latent_dim=latent_dim,
            language_dim=language_dim,
            hidden_dim=hidden_dim,
            num_film_layers=decoder_film_layers,
            use_language=decoder_use_language,
        )

    def encode_action(self, hand_state: torch.Tensor, hand_velocity: torch.Tensor) -> torch.Tensor:
        return self.encoder(hand_state, hand_velocity)

    def emg_to_latent(self, emg_window: torch.Tensor) -> torch.Tensor:
        return self.emg(emg_window)

    def decode(
        self,
        hand_state: torch.Tensor,
        latent: torch.Tensor,
        language_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decoder(hand_state, latent, language_embedding)

    def forward_emg_pipeline(
        self,
        hand_state: torch.Tensor,
        emg_window: torch.Tensor,
        language_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """End-to-end EMG -> latent -> velocity (inference-style)."""
        z = self.emg_to_latent(emg_window)
        return self.decode(hand_state, z, language_embedding)


def blend_synergy_states(
    vision_state: torch.Tensor,
    emg_integrated_state: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    Fixed convex blend (not learned): ``alpha * vision + (1 - alpha) * emg``.

    Args:
        vision_state: (batch, synergy_dim) or (synergy_dim,)
        emg_integrated_state: same shape
        alpha: scalar, (batch, 1), or (batch,) in [0, 1]; visibility / trust in vision
    """
    if alpha.dim() == 1:
        alpha = alpha.unsqueeze(-1)
    return alpha * vision_state + (1.0 - alpha) * emg_integrated_state
