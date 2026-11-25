//! Ruleset & tournament overlay configuration separating Pokemon Showdown
//! `[Gen 2] NC 2000` (`Ruleset::PsStadium2`, default, 100% bit-exact parity)
//! from Gym Leader's Castle PBS `PS.cgi` (`Ruleset::CastlePbs`).
//!
//! 2-layer model (see `pbs-vs-ps-spec.md` §6-1):
//! - Layer 1 (`Ruleset::CastlePbs`): Faithful `PS.cgi` software mechanics.
//! - Layer 2 (`CastleOverlayConfig`): Tournament / player-agreed conventions
//!   (1v1 recoil tie-break, last-mon Perish/Destiny Bond forfeit, turn limit,
//!   `ovfl` overflow mode, and JP event learnset legality).

use serde::{Deserialize, Serialize};

/// Learnset / legality policy for team validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum LearnsetPolicy {
    /// Pokemon Showdown `[Gen 2] NC 2000` (`gen2stadium2`) legality.
    #[default]
    PsGen2Stadium2,
    /// Gym Leader's Castle JP event legality (NYPC 209 moves excluded,
    /// JP domestic event moves allowed, `moveBlackList.csv` min-level check,
    /// and `ev_rules.txt` effort constraints).
    CastleJpEvents,
}

/// Layer 2 player-agreed / room-configured overlay on top of faithful PBS `PS.cgi`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CastleOverlayConfig {
    /// 1v1 recoil double-KO winner: `true` = attacker wins (tournament bot convention),
    /// `false` = raw `PS.cgi` draw.
    pub recoil_1v1_attacker_wins: bool,
    /// Last-mon `Perish Song` / `Destiny Bond` forfeit rule (`rules_2000.txt` L63).
    pub perish_dbond_lastmon_forfeit: bool,
    /// Optional turn-limit judge (tournament NC2000 = `Some(80)` judging alive count
    /// then sum of alive `hp / maxhp` ratios; free/raw = `None`).
    pub turn_limit: Option<u16>,
    /// `PS.cgi` `ovfl` room flag (`false` = standard 999 clamp; `true` = `% 1024`
    /// wrap + missed OHKO instant-KO Counter).
    pub overflow_mode: bool,
    /// Team legality / learnset policy.
    pub learnset_policy: LearnsetPolicy,
}

impl Default for CastleOverlayConfig {
    fn default() -> Self {
        Self::free()
    }
}

impl CastleOverlayConfig {
    /// Standard NC2000 tournament Castle configuration (80-turn judge, tournament overlays).
    #[inline]
    pub const fn tournament() -> Self {
        Self {
            recoil_1v1_attacker_wins: true,
            perish_dbond_lastmon_forfeit: true,
            turn_limit: Some(80),
            overflow_mode: false,
            learnset_policy: LearnsetPolicy::CastleJpEvents,
        }
    }

    /// Raw `PS.cgi` replay configuration (no tournament judge / no 1v1 recoil override).
    #[inline]
    pub const fn raw_cgi(overflow_mode: bool) -> Self {
        Self {
            recoil_1v1_attacker_wins: false,
            perish_dbond_lastmon_forfeit: false,
            turn_limit: None,
            overflow_mode,
            learnset_policy: LearnsetPolicy::CastleJpEvents,
        }
    }

    /// Free-play Castle configuration (no turn limit, standard gentleman overlays).
    #[inline]
    pub const fn free() -> Self {
        Self {
            recoil_1v1_attacker_wins: true,
            perish_dbond_lastmon_forfeit: true,
            turn_limit: None,
            overflow_mode: false,
            learnset_policy: LearnsetPolicy::CastleJpEvents,
        }
    }
}

/// Top-level battle ruleset selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Ruleset {
    /// Pokemon Showdown `[Gen 2] NC 2000` (`gen2stadium2`) — default, bit-exact PS parity.
    #[default]
    PsStadium2,
    /// Gym Leader's Castle PBS `PS.cgi` mechanics with Layer 2 overlay.
    CastlePbs(CastleOverlayConfig),
}

impl Ruleset {
    /// Standard NC2000 tournament Castle ruleset (`L_TURN = 80`).
    #[inline]
    pub const fn castle_tournament() -> Self {
        Self::CastlePbs(CastleOverlayConfig::tournament())
    }

    /// Raw `PS.cgi` ruleset (for exact server probe / raw log verification).
    #[inline]
    pub const fn castle_raw_cgi(overflow_mode: bool) -> Self {
        Self::CastlePbs(CastleOverlayConfig::raw_cgi(overflow_mode))
    }

    /// Free-play Castle ruleset (no turn limit).
    #[inline]
    pub const fn castle_free() -> Self {
        Self::CastlePbs(CastleOverlayConfig::free())
    }

    #[inline]
    pub const fn is_castle(self) -> bool {
        matches!(self, Self::CastlePbs(_))
    }

    #[inline]
    pub const fn is_ps(self) -> bool {
        matches!(self, Self::PsStadium2)
    }

    #[inline]
    pub const fn castle_overlay(self) -> Option<CastleOverlayConfig> {
        match self {
            Self::PsStadium2 => None,
            Self::CastlePbs(cfg) => Some(cfg),
        }
    }

    /// Turn limit after which the battle ends automatically (`Some(1000)` on PS,
    /// `cfg.turn_limit` on Castle).
    #[inline]
    pub const fn turn_limit(self) -> Option<u16> {
        match self {
            Self::PsStadium2 => Some(1000),
            Self::CastlePbs(cfg) => cfg.turn_limit,
        }
    }

    /// Whether `PS.cgi` `ovfl=1` mode is active.
    #[inline]
    pub const fn overflow_mode(self) -> bool {
        match self {
            Self::PsStadium2 => false,
            Self::CastlePbs(cfg) => cfg.overflow_mode,
        }
    }

    /// Whether each side rolls Quick Claw independently (`true` in Castle, `false` on PS).
    #[inline]
    pub const fn independent_quick_claw(self) -> bool {
        self.is_castle()
    }

    /// Whether all EOT phases run in fixed `1P -> 2P` order (`true` in Castle, speed order on PS).
    #[inline]
    pub const fn fixed_eot_order(self) -> bool {
        self.is_castle()
    }

    /// Whether the ruleset satisfies exact P1/P2 seat antisymmetry `V(P1, P2) = -V(P2, P1)`.
    /// False in Castle because EOT phases run in fixed `1P -> 2P` order.
    #[inline]
    pub const fn is_seat_antisymmetric(self) -> bool {
        self.is_ps()
    }

    #[inline]
    pub const fn learnset_policy(self) -> LearnsetPolicy {
        match self {
            Self::PsStadium2 => LearnsetPolicy::PsGen2Stadium2,
            Self::CastlePbs(cfg) => cfg.learnset_policy,
        }
    }

    /// Compact deterministic 64-bit tag for cache/transposition-table key separation.
    /// Returns `0` for `PsStadium2` so PS keys remain bit-identical when folded conditionally.
    #[inline]
    pub const fn fingerprint_tag(self) -> u64 {
        match self {
            Self::PsStadium2 => 0,
            Self::CastlePbs(cfg) => {
                let mut bits: u64 = 0x4353_544C_0000_0000; // "CSTL"
                if cfg.recoil_1v1_attacker_wins {
                    bits |= 1 << 0;
                }
                if cfg.perish_dbond_lastmon_forfeit {
                    bits |= 1 << 1;
                }
                if cfg.overflow_mode {
                    bits |= 1 << 2;
                }
                if matches!(cfg.learnset_policy, LearnsetPolicy::CastleJpEvents) {
                    bits |= 1 << 3;
                }
                if let Some(lim) = cfg.turn_limit {
                    bits |= 1 << 4;
                    bits |= (lim as u64) << 16;
                }
                bits
            }
        }
    }

    /// Parse canonical CLI / WASM / replay string identifiers.
    pub fn parse_id(s: &str) -> Option<Self> {
        let lower = s.trim().to_ascii_lowercase();
        match lower.as_str() {
            "ps" | "ps-stadium2" | "gen2stadium2" | "gen2stadium2nc2000" => Some(Self::PsStadium2),
            "castle" | "castle-pbs" | "castle-free" | "pbs" | "pbs-nc2000" | "pbs-free" => {
                Some(Self::castle_free())
            }
            "castle-tournament" => Some(Self::castle_tournament()),
            "castle-raw" | "castle-cgi" | "pbs-raw" => Some(Self::castle_raw_cgi(false)),
            "castle-raw-ovfl" | "castle-cgi-ovfl" | "pbs-raw-ovfl" => {
                Some(Self::castle_raw_cgi(true))
            }
            _ => lower
                .strip_prefix("castle-custom-")
                .and_then(parse_castle_custom),
        }
    }

    /// Canonical short identifier string.
    pub fn canonical_id(self) -> String {
        match self {
            Self::PsStadium2 => "gen2stadium2".to_string(),
            Self::CastlePbs(cfg) if cfg == CastleOverlayConfig::tournament() => {
                "castle-tournament".to_string()
            }
            Self::CastlePbs(cfg) if cfg == CastleOverlayConfig::raw_cgi(false) => {
                "castle-raw".to_string()
            }
            Self::CastlePbs(cfg) if cfg == CastleOverlayConfig::raw_cgi(true) => {
                "castle-raw-ovfl".to_string()
            }
            Self::CastlePbs(cfg) if cfg == CastleOverlayConfig::free() => {
                "castle-free".to_string()
            }
            Self::CastlePbs(cfg) => format!(
                "castle-custom-r{}-p{}-t{}-o{}-l{}",
                cfg.recoil_1v1_attacker_wins as u8,
                cfg.perish_dbond_lastmon_forfeit as u8,
                cfg.turn_limit.map_or_else(|| "none".to_string(), |t| t.to_string()),
                cfg.overflow_mode as u8,
                match cfg.learnset_policy {
                    LearnsetPolicy::CastleJpEvents => "jp",
                    LearnsetPolicy::PsGen2Stadium2 => "ps",
                },
            ),
        }
    }
}

fn parse_castle_custom(rest: &str) -> Option<Ruleset> {
    let parts: Vec<&str> = rest.split('-').collect();
    if !(4..=5).contains(&parts.len()) {
        return None;
    }
    let parse_bit = |s: &str, prefix: char| -> Option<bool> {
        match s.strip_prefix(prefix)? {
            "0" => Some(false),
            "1" => Some(true),
            _ => None,
        }
    };
    let recoil_1v1_attacker_wins = parse_bit(parts[0], 'r')?;
    let perish_dbond_lastmon_forfeit = parse_bit(parts[1], 'p')?;
    let turn_limit = match parts[2].strip_prefix('t')? {
        "none" => None,
        num => Some(num.parse::<u16>().ok()?),
    };
    let overflow_mode = parse_bit(parts[3], 'o')?;
    let learnset_policy = match parts.get(4).copied() {
        None => LearnsetPolicy::CastleJpEvents,
        Some("ljp") => LearnsetPolicy::CastleJpEvents,
        Some("lps") => LearnsetPolicy::PsGen2Stadium2,
        Some(_) => return None,
    };
    Some(Ruleset::CastlePbs(CastleOverlayConfig {
        recoil_1v1_attacker_wins,
        perish_dbond_lastmon_forfeit,
        turn_limit,
        overflow_mode,
        learnset_policy,
    }))
}
