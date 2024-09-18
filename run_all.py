import copy
import pathlib

import numpy as np
from aiida import engine, orm
from aiida_cp2k.calculations import Cp2kCalculation

from . import  cp2k_utils

ALLOWED_PROTOCOLS = ["standard"]

class Cp2kBenchmarkWorkChain(engine.WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=orm.Code)
        spec.input("structure", valid_type=orm.StructureData)

        spec.input(
            "protocol",
            valid_type=orm.Str,
            default=lambda: orm.Str("standard"),
            required=False,
            help="Either 'standard', ",
        )
        spec.input(
            "wallclock",
            valid_type=orm.Int,
            required=False,
            default=lambda: orm.Int(600),
        )
        spec.input(
            "list_nodes",
            valid_type=orm.List,
            default=lambda: orm.List(list=[]),
            required=True,
            help="List of #nodes to be used in the benchmark.",
        )
        spec.input(
            "list_tasks_per_node",
            valid_type=orm.List,
            default=lambda: orm.List(list=[]),
            required=True,
            help="List of #tasks per node to be used in the benchmark.",
        )
        spec.input(
            "list_threads_per_task",
            valid_type=orm.List,
            default=lambda: orm.List(list=[]),
            required=True,
            help="List of #threads per task to be used in the benchmark.",
        )

        spec.outline(
            cls.setup,
            cls.submit_calculations,
            cls.finalize,
        )
        spec.outputs.dynamic = True

        spec.exit_code(
            381,
            "ERROR_CONVERGENCE1",
            message="SCF of the first step did not converge.",
        )
        spec.exit_code(
            382,
            "ERROR_CONVERGENCE2",
            message="SCF of the second step did not converge.",
        )
        spec.exit_code(
            383,
            "ERROR_NEGATIVE_GAP",
            message="SCF produced a negative gap.",
        )
        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def setup(self):
        self.report("Inspecting input and setting up things")

        self.ctx.files = {
            "basis": orm.SinglefileData(
                file=pathlib.Path(__file__).parent / "data" / "BASIS_MOLOPT"
            ),
            "pseudo": orm.SinglefileData(
                file=pathlib.Path(__file__).parent / "data" / "POTENTIAL"
            ),
            "mpswrapper": orm.SinglefileData(
                file=pathlib.Path(__file__).parent / "data" / "mps-wrapper.sh"
            ),
        }


        return engine.ExitCode(0)


    def submit_calculations(self):
        input_dict = self.ctx.protocol

        for nnodes in self.inputs.list_nodes:
            if nnodes <= 8:
                mywall=50
            else:
	            mywall=20
            # Loop for mpi tasks 
            for ntasks in self.inputs.list_tasks_per_node:
                for nthreads in self.inputs.list_threads_per_task:
                    # Print t only if it is 1 or even
                    if  nthreads<=72/ntasks :
                        # Prepare the builder.
                        builder = Cp2kCalculation.get_builder()
                        builder.code = self.inputs.code
                        builder.structure = self.inputs.structure
                        builder.file = self.ctx.files

                        # Options.
                        builder.metadata.options =  {
                        "max_wallclock_seconds": self.inputs.wallclock.value,
                        "resources": {
                            "num_machines": nnodes,
                            "num_mpiprocs_per_machine": ntasks,
                            "num_cores_per_mpiproc": nthreads,
                        },
                    }

                        builder.metadata.options["parser_name"] = "cp2k_advanced_parser"

                        builder.parameters = orm.Dict(input_dict)
                    
                        submitted_calculation = self.submit(builder)
                        self.report(
                            f"Submitted nodes {nnodes} tasks per node {ntasks} threads {nthreads}: {submitted_calculation.pk}"
                        )
                        self.to_context(
                            **{
                                f"run_{nnodes}_{ntasks}_{nthreads}": engine.append_(
                                    submitted_calculation
                                )
                            }
                        )

    def check_scf(self):
        return (
            engine.ExitCode(0)
            if self.ctx.scf.is_finished_ok
            else self.exit_codes.ERROR_TERMINATION
        )


    def finalize(self):
        self.report("Finalizing...")

        if not self.ctx.second_step.is_finished_ok:
            return self.exit_codes.ERROR_TERMINATION
        if not self.ctx.second_step.outputs.std_output_parameters["motion_step_info"][
            "scf_converged"
        ][-1]:
            self.report("GW step did not converge")
            return self.exit_codes.ERROR_CONVERGENCE2

        self.out(
            "std_output_parameters", self.ctx.second_step.outputs.std_output_parameters
        )
        self.out(
            "gw_output_parameters", self.ctx.second_step.outputs.gw_output_parameters
        )

        return engine.ExitCode(0)
