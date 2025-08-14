import json
import typing

import numpy as np

from ionq.noise_models.Noise import (
    ConstantOffsetNoiseModel,
    NoNoiseModel,
)
from ionq.pulse_shaping.utilities import convert_gate_file_dict_to_solutions
from ionq.pulse_sim.CoupledModel.noise_configs import ConfigCoupledModel
from ionq.pulse_sim.Simulator import ArbitraryMagnus
from ionq.pulse_sim.Simulator.ControlChannel import SystemControl
from ionq.pulse_sim.Simulator.Hamiltonian import (
    BasicHamiltonian,
    HamiltonianLindbladian,
    MotionalDephasingLindbladian,
    MotionalHeatingLindbladian,
    QubitDephasingLindbladian,
)
from ionq.pulse_sim.Simulator.PulseUtils import (
    average_gate_fidelity,
    average_gate_fidelity_molmer,
)
from ionq.pulses.MSPulse import PieceWiseConstantMSPulse
from ionq.system_parameters import (
    HardwareConfiguration,
    HardwareStateModel,
    SoftwareState,
    SystemType,
)
from ionq.system_parameters.BeamShape import CircularGaussianBeam


class CoupledModel:
    default_kwargs = {
        "amplitude_noise": None,
        "phase_noise": None,
        "motional_dephasing": None,
        "mode_frequency_noise": None,
        "qubit_dephasing_noise": None,
        "thermal_with_displacement": None,
        "heating": None,
        "crosstalk": None,
        "beam_shaping": None,
        "motional_phase": None,
    }
    DefaultConfig = ConfigCoupledModel(**default_kwargs)

    def __init__(
        self,
        param_path,  # "prism_2_ions.json" # json file describing type of ion chain (smaller chain - prism_2_ions)
        gateset_path,
        noise_models: ConfigCoupledModel
        | None = DefaultConfig,  # json file describing type of ion chain (smaller chain - output_2_ions)
        # gateset_path = EXAMPLES_FOLDER / "c1-am-dev-25-06-03-sgauss-100us-max200kHz (1).cfg" # gateset file for the 2 ion chain
        qubit1: int = 0,  # target qubit
        qubit2: int = 1,  # reference (control) qubit
        num_monte_carlo_samples: int = 1,
        system_type=SystemType.Tempo,
        gate_type: str = "MS",  # "POPEYE" or "MS",
        print_progress: bool = False,  # whether to print progress of the simulation
    ):
        self._param_path = param_path
        self._gateset_path = gateset_path
        self._qubit1 = qubit1
        self._qubit2 = qubit2
        self._num_monte_carlo_samples = num_monte_carlo_samples
        self._system_type = system_type
        self.gate_type = gate_type
        self._noise_models = noise_models

        self._construct_hardware_configuration()
        self._construct_software_state()
        self._construct_pulse()
        self._construct_hardware_state_model()
        self._construct_system_control()
        self._construct_solver()
        self._print_progress = print_progress

    def _construct_hardware_configuration(self) -> HardwareConfiguration:
        mode_data = None  # you can add in custom mode data here

        if self._noise_models.beam_shaping is None:  # type: ignore
            beam_shape = None
        else:
            if self._noise_models.beam_shaping.beam_profile == "gaussian":  # type: ignore
                beam_shape = CircularGaussianBeam(self._noise_models.beam_shaping.waist)  # type: ignore
            else:
                raise ValueError(
                    f"Unimplemented beam profile: {self._noise_models.beam_shaping.beam_profile}"  # type: ignore
                )

        self._hardware = HardwareConfiguration.load_from_mode_data_and_system_type(
            prism_file=self._param_path,
            system_type=self._system_type,
            mode_data=mode_data,
            beam_shape=beam_shape,
        )
        return self._hardware

    def _construct_software_state(self) -> SoftwareState:
        hpi = np.pi / 2
        phi_s = hpi
        phi_m = 0.0
        if self._noise_models.motional_phase is not None:  # type: ignore
            phi_m = self._noise_models.motional_phase.phi_m  # type: ignore
        if self._noise_models.spin_phase is not None:  # type: ignore
            phi_s = self._noise_models.spin_phase.phi_s  # type: ignore

        beam_phases = [
            phi_s + phi_m,
            phi_s - phi_m,
            0,
            0,
            phi_s + phi_m,
            phi_s - phi_m,
            0,
            0,
        ]
        self._software_state = SoftwareState.generate_from_hardware(
            self._hardware, phases=beam_phases
        )
        return self._software_state

    def _construct_pulse(self) -> PieceWiseConstantMSPulse:
        pulse_set = convert_gate_file_dict_to_solutions(
            json.load(open(self._gateset_path))
        )
        # Get the specific pulse that we want
        self._pulse: PieceWiseConstantMSPulse = typing.cast(
            PieceWiseConstantMSPulse,
            pulse_set[(self._qubit1, self._qubit2, 0)].convert_to_MSPulse(
                software_state=self._software_state
            ),
        )

        return self._pulse

    def _construct_hardware_state_model(self) -> HardwareStateModel:
        hardware_state_model_config = {
            "mode_frequency_noises": None,
            "mode_frequency_drifts": None,
            "mode_participation_drifts": None,
            "beam_amplitude_relative_noises": None,
            "beam_amplitude_scaling_functions": None,
            "beam_phase_noises": None,
            "beam_phase_drift_functions": None,
            "manifold_variation_functions": None,
            "mode_heating_rates": None,
            "mode_cooling_rates": None,
            "mode_dephasing_rates": None,
            "qubit_dephasing_rates": None,
            "extra_noise_models": None,
        }

        ## Coherent Noise Sources
        if self._noise_models.mode_frequency_noise is not None:  # type: ignore
            # mean = 0
            # std_dev = 250e-6
            delta_nu = np.random.normal(
                self._noise_models.mode_frequency_noise.mean,  # type: ignore
                self._noise_models.mode_frequency_noise.standarddeviation,  # type: ignore
                1,
            )
            mode_frequency_noises = [
                ConstantOffsetNoiseModel(2 * np.pi * delta_nu[0]) for _ in range(2)
            ]

        if self._noise_models.phase_noise is not None:  # type: ignore
            noise_dict = self._noise_models.phase_noise.noise_spectra  # type: ignore
            hardware_state_model_config["beam_amplitude_relative_noises"] = [  # type: ignore
                noise_dict["ion_0_noise_blue"],
                noise_dict["ion_0_noise_red"],
                NoNoiseModel(),
                noise_dict["ion_1_noise_blue"],
                noise_dict["ion_1_noise_red"],
                NoNoiseModel(),
            ]

        if self._noise_models.amplitude_noise is not None:  # type: ignore
            noise_dict = self._noise_models.amplitude_noise.noise_spectra  # type: ignore
            hardware_state_model_config["beam_amplitude_relative_noises"] = [  # type: ignore
                noise_dict["ion_0_noise_blue"],
                noise_dict["ion_0_noise_red"],
                NoNoiseModel(),
                noise_dict["ion_1_noise_blue"],
                noise_dict["ion_1_noise_red"],
                NoNoiseModel(),
            ]

        ## Incoherent Noise Sources
        if self._noise_models.motional_dephasing is not None:  # type: ignore
            hardware_state_model_config["mode_dephasing_rates"] = (
                self._noise_models.motional_dephasing.dephasing_rates
            )  # type: ignore

        if self._noise_models.qubit_dephasing_noise is not None:  # type: ignore
            hardware_state_model_config["qubit_dephasing_rates"] = (
                self._noise_models.qubit_dephasing_noise.dephasing_rates
            )  # type: ignore

        if self._noise_models.heating is not None:  # type: ignore # type: ignores
            hardware_state_model_config["mode_heating_rates"] = (
                self._noise_models.heating.heating_rates
            )  # type: ignore
            hardware_state_model_config["mode_cooling_rates"] = (
                self._noise_models.heating.heating_rates
            )  # type: ignore

        ## Construct the hardware state model
        self._hardware_state = HardwareStateModel(
            self._hardware, **hardware_state_model_config
        )

        return self._hardware_state

    def _construct_system_control(self) -> SystemControl:
        match self.gate_type:
            case "MS":
                self._system_control = SystemControl(
                    self._pulse,
                    self._hardware,
                    self._hardware_state,
                    self._software_state,
                    ions_in_simulation=[self._qubit1, self._qubit2],
                    modes_in_simulation=[0, 1],
                    beam_ion_interactions_simulated=[
                        (0, self._qubit1),
                        (1, self._qubit1),
                        (2, self._qubit1),
                        (4, self._qubit2),
                        (5, self._qubit2),
                        (6, self._qubit2),
                    ],
                )
                if self._noise_models.crosstalk is not None:  # type: ignore
                    raise ValueError(
                        "Crosstalk noise model is not implemented yet for MS system type."
                    )
            case "POPEYE":
                if self._noise_models.crosstalk is not None:  # type: ignore
                    self._system_control = SystemControl(
                        self._pulse,
                        self._hardware,
                        self._hardware_state,
                        self._software_state,
                        ions_in_simulation=[self._qubit1, self._qubit2],
                        modes_in_simulation=[0, 1],
                        beam_ion_interactions_simulated=[
                            (0, self._qubit1),
                            (1, self._qubit1),
                            (2, self._qubit1),
                            (0, self._qubit2),
                            (1, self._qubit2),
                            (2, self._qubit2),
                        ],
                    )
                else:
                    self._system_control = SystemControl(
                        self._pulse,
                        self._hardware,
                        self._hardware_state,
                        self._software_state,
                        ions_in_simulation=[self._qubit1],
                        beam_ion_interactions_simulated=[
                            (0, self._qubit1),
                            (1, self._qubit1),
                            (2, self._qubit1),
                        ],
                    )
            case _:
                raise ValueError("gate_type must be either 'POPEYE' or 'MS'")

        return self._system_control

    def _construct_solver(
        self,
    ) -> ArbitraryMagnus.ArbitraryMagnus | ArbitraryMagnus.ArbitraryMagnusOpen:
        hamiltonian = BasicHamiltonian(self._system_control)
        lindbladian = None

        solver_kwargs = {
            "number_number_states": 50,
            "method": "RK45",
            "relative_tolerance": 1e-8,
            "absolute_tolerance": 1e-8,
            "term_elimination_threshold": 1e-9,
            "max_order": 3,
        }

        if self._noise_models.thermal_with_displacement is not None:  # type: ignore
            thermal_with_displacement_config = (
                self._noise_models.thermal_with_displacement
            )  # type: ignore
            n_bar = thermal_with_displacement_config.nbars
            displacements = thermal_with_displacement_config.displacements
            avg_n = n_bar + np.power(np.abs(displacements), 2)
            print(
                "Warning: Updating num_number_states based on average n is a work in progress"
            )

        if (
            self._noise_models.motional_dephasing is not None  # type: ignore
            or self._noise_models.heating is not None  # type: ignore
            or self._noise_models.qubit_dephasing_noise is not None  # type: ignore
        ):  # type: ignore
            lindbladian = HamiltonianLindbladian(self._system_control, hamiltonian)
        if self._noise_models.motional_dephasing is not None:  # type: ignore
            lindbladian = lindbladian + MotionalDephasingLindbladian(
                self._system_control
            )  # type: ignore

        if self._noise_models.heating is not None:  # type: ignore
            lindbladian = lindbladian + MotionalHeatingLindbladian(
                self._system_control, fermion_space_map=[2, 2]
            )  # type: ignore

        if self._noise_models.qubit_dephasing_noise is not None:  # type: ignore
            lindbladian = lindbladian + QubitDephasingLindbladian(self._system_control)  # type: ignore

        for keys in self._noise_models.__dict__:
            if self._noise_models.__dict__[keys] is not None:
                if (
                    self._noise_models.__dict__[keys].minMagnus
                    > solver_kwargs["max_order"]
                ):
                    solver_kwargs["max_order"] = self._noise_models.__dict__[
                        keys
                    ].minMagnus

        if lindbladian is None:
            self._simulator = ArbitraryMagnus.ArbitraryMagnus(
                hamiltonian, self._system_control, **solver_kwargs
            )
        else:
            self._simulator = ArbitraryMagnus.ArbitraryMagnusOpen(
                lindbladian, self._system_control, **solver_kwargs
            )

        return self._simulator

    def run_simulation(self):
        infid = 0
        match self.gate_type:
            case "MS":
                target_unitary = self._pulse.get_target_unitary()
            case "POPEYE":
                if self._noise_models.crosstalk is not None:
                    target_unitary = np.eye(4)
                else:
                    target_unitary = np.eye(2)
            case _:
                raise ValueError("Gate Type must be either 'POPEYE' or 'MS'")

        if self._noise_models.thermal_with_displacement is not None:
            thermal_with_displacement_config = (
                self._noise_models.thermal_with_displacement
            )
            n_bar = thermal_with_displacement_config.nbars
            displacements = thermal_with_displacement_config.displacements
            superop_kwargs = {
                "average_mode_occupations": n_bar,
                "alphas": displacements,
                "monte_carlo_samples": self._num_monte_carlo_samples,
                "print_progress": False,
                "use_parallel": False,
                "max_time_secs": None,
            }
            super_op = (
                self._simulator.calculate_superoperator_from_pulse_with_displacement(
                    **superop_kwargs
                )
            )
            kraus = self._simulator.calculate_kraus_ops_from_superoperator_matrix(
                super_op
            )
            infid = (1.0 - average_gate_fidelity_molmer(kraus, target_unitary)) * 1e4
            if self._print_progress:
                print("Infidelity(pptt)\t", infid)

        else:
            n_bar = [0.0, 0.0]
            superop_kwargs = {
                "average_mode_occupations": n_bar,
                "monte_carlo_samples": self._num_monte_carlo_samples,
                "print_progress": False,
                "use_parallel": False,
                "max_time_secs": None,
            }

            super_op = self._simulator.calculate_superoperator_from_pulse(
                **superop_kwargs
            )
            kraus = self._simulator.calculate_kraus_ops_from_superoperator_matrix(
                super_op
            )
            infid = (1.0 - average_gate_fidelity(kraus, target_unitary)) * 1e4
            if self._print_progress:
                print("Infidelity(pptt)\t", infid)

        return infid, kraus, super_op, self._simulator

    def list_of_noise_models(self):
        print("List of noise models Implemented So far:")
        for key in self.default_kwargs:
            print(f"\t {key}")
        print("Cross talk noise model is not implemented yet for MS system type.")

        print("List of noise models to be added in future:")
        print("\t light shift")
        print("\t axial coupling")
        print("\t stray electric fields")
        print("\t spontaneous emission")
        print("\t motional mode drift")
        print("\t motional mode noise from PSD")
        print("\t cross talk for MS")

    def list_of_gates(self):
        print("List of gates in the gateset:")
        print("\t MS")
        print("\t POPEYE")
