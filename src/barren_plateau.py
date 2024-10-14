import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np

np.random.seed(42)


class BarrenPlateau:
    def __init__(
        self,
        nqubits_list: list[int],
        nlayers_list: list[int],
        ansatz_type: str,
        nsamples: int = 3,
        shots: int = None,
        observable = None  # use a global observable if None
    ):

        self.nqubits_list = nqubits_list
        self.nlayers_list = nlayers_list
        self.ansatz_type = ansatz_type
        self.nsamples = nsamples
        self.shots = shots
        self.observable = observable
        self.nobs = None
        
        valid_ansatz_types = {"TPA", "HEA", "SEA"}
        if self.ansatz_type not in valid_ansatz_types:
            raise ValueError(f"ansatz_type must be one of {valid_ansatz_types}")

    def make_initial_params(self, nqubits, nlayers):
        """Generate random parameters corresponding to the ansatz_type
        Args:
            nqubits (int): the number of qubits in the circuit
            nlayers (int): the number of layers of ansatz
        Returns:
            array[float]: array of parameters
        """
        if self.ansatz_type == "TPA":
            params = np.random.uniform(0, np.pi, size=nqubits * nlayers)
        elif self.ansatz_type == "HEA":
            params = np.random.uniform(0, np.pi, size=nqubits * nlayers)
        else:
            shape = qml.StronglyEntanglingLayers.shape(nlayers, n_wires=nqubits)
            params = np.random.random(size=shape)

        return params
    
    @staticmethod
    def make_global_observable(nqubits):
        """Generate the global observable H = |0><0|^{\otimes n}"""
        H = np.zeros((2**nqubits, 2**nqubits))
        H[0, 0] = 1

        wirelist = [i for i in range(nqubits)]

        return qml.Hermitian(H, wirelist)

    def ansatz(self, params, nqubits, nlayers):
        """Generate the variational quantum circuit.
        Args:
            params (array[float]): array of parameters
            nqubits (int): the number of qubits in the circuit
            nlayers (int): the number of layers of ansatz
        Returns:
            expectation value (float): the expectation value of the target observable
        """

        if self.ansatz_type == "TPA":
            for i in range(nlayers):
                for j in range(nqubits):
                    qml.RX(params[nqubits * i + j], wires=j)
                    qml.RY(params[nqubits * i + j], wires=j)
        elif self.ansatz_type == "HEA":
            for i in range(nlayers):
                for j in range(nqubits):
                    qml.RX(params[nqubits * i + j], wires=j)
                    qml.RY(params[nqubits * i + j], wires=j)
                for j in range(nqubits - 1):
                    qml.CNOT(wires=[j, j + 1])
        else:
            qml.StronglyEntanglingLayers(params, wires=range(nqubits))

    def make_circuit(self, nqubits):
        """Generate a variational quantum circuit
        Args:
            nqubits (int): the number of qubits in the circuit
        Returns:
            QuantumCircuit: variational quantum circuit
        """
        dev = qml.device("lightning.qubit", wires=nqubits)

        def func(params, nqubits, nlayers):

            self.ansatz(params, nqubits, nlayers)

            if self.observable is None:
                observable = self.make_global_observable(nqubits)
            else:
                observable = self.observable

            return qml.expval(observable)

        qcircuit = qml.QNode(func, dev, diff_method="adjoint")
        return qcircuit

    def bp_nlayers(self):
        """Simulate barren plateaus. Var<dH/dtheta_i> vs the number of layers of ansatz.
        Returns:
            means, vars ([[float],..]): 2d-list of means and variances of gradients for each nlayers and nqubits.
        """
        self.grad_means_list = []
        self.grad_vars_list = []

        for nqubits in self.nqubits_list:
            grad_means = []
            grad_vars = []

            for nlayers in self.nlayers_list:
                gradients_list = []
                qcircuit = self.make_circuit(nqubits)
                grad = qml.grad(qcircuit, argnum=0)

                for _ in range(self.nsamples):
                    params = self.make_initial_params(nqubits, nlayers)
                    gradients = grad(params, nqubits, nlayers)
                    gradients_list.append(gradients)

                # By transpose, each list in gradients_list have nsample gradients of one parameter.
                gradients_list = np.array(gradients_list).T

                if self.observable is None:
                    self.nobs = nqubits
                else:
                    self.nobs = len(list(self.observable.wires))

                # take an average of variances of the parameter gradients on which the cost depends.
                # When we use 'TPA', we exclude the parameters on which the cost does not depend.
                if self.ansatz_type == "TPA" and self.observable is not None:
                    gradients_list_for_TPA = []
                    for j in range(nlayers):
                        for k in list(self.observable.wires):
                            gradients_list_for_TPA.append(gradients_list[j * nqubits + k])

                    grad_means.append(
                        np.mean(
                            [np.mean(row) for row in gradients_list_for_TPA]
                        )
                    )
                    grad_vars.append(
                        np.mean(
                            [np.var(row) for row in gradients_list_for_TPA]
                        )
                    )
                else:
                    grad_means.append(
                        np.mean(
                            [np.mean(row) for row in gradients_list]
                        )
                    )
                    grad_vars.append(
                        np.mean(
                            [np.var(row) for row in gradients_list]
                        )
                    )

            self.grad_means_list.append(grad_means)
            self.grad_vars_list.append(grad_vars)
            print(f"nqubits {nqubits},  observable size 2^{self.nobs} * 2^{self.nobs}")

        # return self.grad_means_list, self.grad_vars_list

    def plot_bp_nlayers(self, x_axis='nqubits', mean=False):
        if x_axis in ('nqubits', 'nlayers', '3D'):
            pass
        else:
            raise NameError("Input the correct x_axis: 'nqubits', 'nlayers' or '3D'")
        
        if mean:
            for i, nqubits in enumerate(self.nqubits_list):
                plt.plot(self.nlayers_list, self.grad_means_list[i], label=f"nqubits = {nqubits}", marker="o")

            plt.plot(self.nlayers_list, np.zeros(len(self.nlayers_list)), color="black", linestyle='dashed')
            plt.xlabel(r"nlayers", fontsize=18)
            plt.ylabel(r"$\langle \partial_{\theta_i} E \rangle$ mean", fontsize=18)
            plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
            plt.show()
        else:
            pass

        if x_axis == 'nqubits':
            for j, nlayers in enumerate(self.nlayers_list):
                plt.semilogy(self.nqubits_list, np.array(self.grad_vars_list).T[j], label=f"nlayers = {nlayers}", marker="o")

            plt.xlabel(r"nqubits", fontsize=18)
            plt.ylabel("$Var[\partial_{\\theta} C]$", fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.ylim([1e-7, 1e-0])
            plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=18)
            plt.title(f"{self.ansatz_type}; {(self.observable is None)*'Global'+(self.observable is not None)*'Local'} Observable", fontsize=18)
            # plt.savefig(f"{self.ansatz_type}-{(self.observable is None)*'global'+(self.observable is not None)*'local'}-ob.pdf", bbox_inches="tight")
            plt.show()
        elif x_axis == 'nlayers':
            for i, nqubits in enumerate(self.nqubits_list):
                plt.semilogy(self.nlayers_list, self.grad_vars_list[i], label=f"nqubits = {nqubits}", marker="o")

            plt.xlabel(r"L", fontsize=18)
            plt.ylabel("$Var[\partial_{\\theta} C]$", fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.ylim([1e-7, 1e-0])
            plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=18)
            plt.title(f"{self.ansatz_type}; {(self.observable is None)*'Global'+(self.observable is not None)*'Local'} Observable", fontsize=18)
            # plt.savefig(f"{self.ansatz_type}-{(self.observable is None)*'global'+(self.observable is not None)*'local'}-ob.pdf", bbox_inches="tight")
            plt.show()
        else:
            ## 3D plot
            mesh_nqubits, mesh_nlayers = np.meshgrid(self.nqubits_list, self.nlayers_list)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(mesh_nqubits, mesh_nlayers, np.log(np.array(self.grad_vars_list).T), cmap='viridis')
            ax.set_xlabel('nqubits')
            ax.set_ylabel('nlayers')
            ax.set_zlabel('Var[grad]')
            ax.view_init(40, 40)
            ax.set_box_aspect(None, zoom=0.8)
            plt.show()

    # nlayers_list must contain one element.
    def bp_nqubits(self):
        """Simulate barren plateaus. Var<dH/dtheta_i> vs the number of qubits.
            This function is defined by bp_nlayers().
        Returns:
            means, variances (list[float]): list of means and variances of gradients for each nqubits
        """
        self.bp_nlayers()
        # return self.grad_means, self.grad_vars

    def plot_bp_nqubits(self, mean=False):
        if mean:
            plt.plot(self.nqubits_list, self.grad_means_list, label="mean", marker="o")
            plt.plot(self.nqubits_list, np.zeros(len(self.nqubits_list)))
            plt.xlabel(r"n", fontsize=18)
            plt.ylabel(r"$\langle \partial_{\theta_i} E \rangle$", fontsize=18)
            plt.legend()
            plt.show()
        else:
            pass

        # Fit the semilog plot to a straight line
        p = np.polyfit(
            x=self.nqubits_list,
            y=np.log(
                np.clip(a=np.ravel(self.grad_vars_list), a_min=1e-323, a_max=None)
            ),
            deg=1,
        )

        # Plot the straight line fit to the semilog
        plt.semilogy(self.nqubits_list, self.grad_vars_list, marker="o")
        plt.semilogy(
            self.nqubits_list,
            np.exp(p[0] * self.nqubits_list + p[1]),
            label="Slope {:3.2f}".format(p[0]),
            linestyle="dashed",
        )
        
        plt.tick_params(labelsize=18)
        plt.xlabel(r"N Qubits", fontsize=18)
        plt.ylabel("$Var[\partial_{\\theta} C]$", fontsize=18)
        plt.legend()
        plt.show()
