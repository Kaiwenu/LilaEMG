"""
LILA-style latent action model for EMG teleoperation (structure aligned with ``lila/src/models/film.py``).

Components:
  - ActionEncoder E_a: (hand_state, hand_velocity) + language -> latent via FiLM on hidden (GELU MLPs).
  - EMGNetwork f_emg: EMG window / features -> latent (same space as encoder; ReLU MLP — joystick replacement).
  - ActionDecoder D: (hand_state, latent) + language -> synergy velocity via FiLM on hidden (GELU MLPs).

Training stages are implemented in ``train_teleop.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionEncoder(nn.Module):
    """
    FiLM-GeLU encoder matching ``lila.src.models.film.FiLM`` encoder arm:

    ``enc2film``: (hand_state ‖ hand_velocity) -> hidden, then language generates (gamma, beta) via
    ``enc_film_gen`` + ``efg`` / ``efb``, FiLM on hidden, then ``film2latent`` -> latent.
    """

    def __init__(
        self,
        synergy_dim: int,
        latent_dim: int = 2,
        language_dim: int = 768,
        hidden_dim: int = 256,
        use_language: bool = True,
    ):
        super().__init__()
        self.synergy_dim = synergy_dim
        self.latent_dim = latent_dim
        self.language_dim = language_dim
        self.hidden_dim = hidden_dim
        self.use_language = use_language

        in_dim = 2 * synergy_dim
        self.enc_film_gen = nn.Sequential(
            nn.Linear(language_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.efg = nn.Linear(hidden_dim, hidden_dim)
        self.efb = nn.Linear(hidden_dim, hidden_dim)

        self.enc2film = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.film2latent = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        hand_state: torch.Tensor,
        hand_velocity: torch.Tensor,
        language_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hand_state: (batch, synergy_dim)
            hand_velocity: (batch, synergy_dim)
            language_embedding: (batch, language_dim); required if ``use_language``
        Returns:
            latent: (batch, latent_dim)
        """
        if self.use_language and language_embedding is None:
            raise ValueError("language_embedding is required when use_language=True")
        x = torch.cat([hand_state, hand_velocity], dim=-1)
        to_film = self.enc2film(x)
        if self.use_language:
            film_emb = self.enc_film_gen(language_embedding)  # type: ignore[arg-type]
            gamma, beta = self.efg(film_emb), self.efb(film_emb)
            h = gamma * to_film + beta
        else:
            h = to_film
        return self.film2latent(h)


class EMGNetwork(nn.Module):
    """
    f_emg: EMG window flattened -> latent (same space as ActionEncoder). Unchanged (joystick replacement).
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
    FiLM-GeLU decoder matching ``lila.src.models.film.FiLM`` decoder arm:

    ``dec2film``: (hand_state ‖ latent) -> hidden, then language FiLM via ``dec_film_gen`` + ``dfg`` / ``dfb``,
    then ``film2action`` -> synergy velocity.
    """

    def __init__(
        self,
        synergy_dim: int,
        latent_dim: int = 2,
        language_dim: int = 768,
        hidden_dim: int = 256,
        use_language: bool = True,
    ):
        super().__init__()
        self.synergy_dim = synergy_dim
        self.latent_dim = latent_dim
        self.language_dim = language_dim
        self.hidden_dim = hidden_dim
        self.use_language = use_language

        self.dec2film = nn.Sequential(
            nn.Linear(synergy_dim + latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dec_film_gen = nn.Sequential(
            nn.Linear(language_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.dfg = nn.Linear(hidden_dim, hidden_dim)
        self.dfb = nn.Linear(hidden_dim, hidden_dim)
        self.film2action = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, synergy_dim),
        )

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
        if self.use_language and language_embedding is None:
            raise ValueError("language_embedding is required when use_language=True")
        y = torch.cat([hand_state, latent], dim=-1)
        to_film = self.dec2film(y)
        if self.use_language:
            film_emb = self.dec_film_gen(language_embedding)  # type: ignore[arg-type]
            gamma, beta = self.dfg(film_emb), self.dfb(film_emb)
            h = gamma * to_film + beta
        else:
            h = to_film
        return self.film2action(h)


class LilaTeleopModel(nn.Module):
    """
    Full graph: ``encoder``, ``emg``, ``decoder``.

    Typical forwards:
      - Stage 1: z = encoder(s, v, lang); v_hat = decoder(s, z, lang)
      - Stage 2: z_star = encoder(s, v, lang); z_emg = emg(emg_window); loss = ||z_emg - z_star||
      - Inference (EMG path): z = emg(emg_window); v_hat = decoder(s, z, lang)
    """

    def __init__(
        self,
        synergy_dim: int,
        latent_dim: int = 2,
        emg_dim: int = 320,
        language_dim: int = 768,
        hidden_dim: int = 256,
        emg_hidden_layers: int = 2,
        encoder_use_language: bool = True,
        decoder_use_language: bool = True,
    ):
        super().__init__()
        self.synergy_dim = synergy_dim
        self.latent_dim = latent_dim
        self.emg_dim = emg_dim
        self.language_dim = language_dim

        self.encoder = ActionEncoder(
            synergy_dim=synergy_dim,
            latent_dim=latent_dim,
            language_dim=language_dim,
            hidden_dim=hidden_dim,
            use_language=encoder_use_language,
        )
        self.emg = EMGNetwork(
            emg_dim=emg_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_hidden=emg_hidden_layers,
        )
        self.decoder = ActionDecoder(
            synergy_dim=synergy_dim,
            latent_dim=latent_dim,
            language_dim=language_dim,
            hidden_dim=hidden_dim,
            use_language=decoder_use_language,
        )

    def encode_action(
        self,
        hand_state: torch.Tensor,
        hand_velocity: torch.Tensor,
        language_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(hand_state, hand_velocity, language_embedding)

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
