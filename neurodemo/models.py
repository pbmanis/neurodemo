# -*- coding: utf-8 -*-
"""
Model registry for neurodemo.

Each entry in MODELS describes a simulation preset: which channels are
active (and visible in the parameter tree), what temperature/speed to use,
and which DemoWindow method sets default reversal potentials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Type

import neurodemo.units as NU
from neurodemo.channels import (
    CaL, CaT, HHK, HHNa, IH, KA, Leak, LGKfast, LGKslow, LGNa,
    MHNa, MHK, MHCaT, MHIh,
)


@dataclass
class ChannelSpec:
    """Configuration for one channel within a ModelConfig."""
    cls: Type
    enabled: bool = True
    defaults: dict = field(default_factory=dict)
    # 'defaults' keys match ChannelParameter child names: 'Erev', 'Gmax'


@dataclass
class ModelConfig:
    """Complete description of a simulation preset."""
    name: str
    temp: float
    speed: float
    channels: list          # list[ChannelSpec] — only channels that should be VISIBLE
    cap: float = 10e-12     # cell capacitance in F (default 10 pF matches Section default)
    setup_erev: Optional[str] = None  # DemoWindow method name to call after load


# Ordered list of every channel class the main window instantiates.
# Channels absent from a given ModelConfig's list will be hidden/disabled.
ALL_CHANNEL_CLASSES = [
    Leak, HHNa, HHK, IH, KA, CaL, CaT,
    LGNa, LGKfast, LGKslow,
    MHNa, MHK, MHCaT, MHIh,
]


# Thalamic TC cell area derived from cap / cap_bar (1 µF/cm² specific capacitance).
# Used to convert gbar (mS/cm²) → Gmax (S) for preset defaults.
_TC_AREA = 0.29e-9 / (NU.uF / NU.cm ** 2)   # ≈ 2.9e-8 m²
_tc_gmax = lambda g: g * NU.mS / NU.cm ** 2 * _TC_AREA


MODELS: dict[str, ModelConfig] = {
    'Passive': ModelConfig(
        name='Passive',
        temp=6.3,
        speed=1.0,
        setup_erev=None,
        channels=[
            ChannelSpec(Leak, defaults={'Gmax': 1 * NU.nS, 'Erev': 0.0}),
        ],
    ),
    'HH AP': ModelConfig(
        name='HH AP',
        temp=6.3,
        speed=1.0,
        setup_erev='set_hh_erev',
        channels=[
            ChannelSpec(Leak, defaults={'Gmax': 1 * NU.nS, 'Erev': -55 * NU.mV}),
            ChannelSpec(HHNa, defaults={'Erev': 50 * NU.mV}),
            ChannelSpec(HHK, defaults={'Erev': -74 * NU.mV}),
        ],
    ),
    'Extended HH': ModelConfig(
        name='Extended HH',
        temp=6.3,
        speed=1.0,
        setup_erev='set_hh_erev',
        channels=[
            ChannelSpec(Leak, defaults={'Gmax': 1 * NU.nS, 'Erev': -55 * NU.mV}),
            ChannelSpec(HHNa, defaults={'Erev': 50 * NU.mV}),
            ChannelSpec(HHK, defaults={'Erev': -74 * NU.mV}),
            ChannelSpec(IH, enabled=False, defaults={'Erev': -43 * NU.mV}),
            ChannelSpec(KA, enabled=False, defaults={'Erev': -74 * NU.mV}),
            ChannelSpec(CaL, enabled=False, defaults={'Erev': 140 * NU.mV}),
            ChannelSpec(CaT, enabled=False, defaults={'Erev': 140 * NU.mV}),
        ],
    ),
    'LG AP': ModelConfig(
        name='LG AP',
        temp=37.0,
        speed=1.0,
        setup_erev='set_lg_erev',
        channels=[
            ChannelSpec(Leak, defaults={'Gmax': 2.5 * NU.nS, 'Erev': -70 * NU.mV}),
            ChannelSpec(LGNa, defaults={'Erev': 74 * NU.mV}),
            ChannelSpec(LGKfast, defaults={'Erev': -90 * NU.mV}),
            ChannelSpec(LGKslow, defaults={'Erev': -90 * NU.mV}),
        ],
    ),
    'Thalamic TC': ModelConfig(
        name='Thalamic TC',
        temp=35.5,
        speed=1.0,
        cap=0.29e-9,
        setup_erev=None,
        channels=[
            # All Gmax = gbar (mS/cm²) × _TC_AREA set explicitly so the GUI
            # displays correct values and the simulation is area-independent.
            # Leak gbar raised to 0.12 mS/cm² (≈35 nS) to balance the ~175 pA
            # inward window current from Na/CaT/Ih at the -65 mV resting potential.
            ChannelSpec(Leak,  defaults={'Gmax': _tc_gmax(0.12), 'Erev': -70 * NU.mV}),
            ChannelSpec(MHNa,  defaults={'Gmax': _tc_gmax(90),   'Erev':  50 * NU.mV}),
            ChannelSpec(MHK,   defaults={'Gmax': _tc_gmax(10),   'Erev': -90 * NU.mV}),
            ChannelSpec(MHCaT, defaults={'Gmax': _tc_gmax(2.0),  'Erev': 140 * NU.mV}),
            ChannelSpec(MHIh,  defaults={'Gmax': _tc_gmax(0.05), 'Erev': -43 * NU.mV}),
        ],
    ),
}
