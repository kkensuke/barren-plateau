import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np

np.random.seed(42)


class BarrenPlateau:
    def __init__(
        self,
        nqubits_list: list[int],
        nlayers_list: list[int],
        initialization_method: str,
        nsamples: int = 3,
        shots: int = None,
        observable = None  # use a global observable if None
    ):

        self.nqubits_list = nqubits_list
        self.nlayers_list = nlayers_list
        self.initialization_method = initialization_method
        self.nsamples = nsamples
        self.shots = shots
        self.observable = observable
        self.obs_locality = None
        
        valid_initialization_types = {"Small", "MBL", "Random", "Gaussian"}
        if self.initialization_method not in valid_initialization_types:
            raise ValueError(f"Invalid initialization type. Must be one of {valid_initialization_types}")

    def make_initial_params(self, nqubits, nlayers, obs_locality) -> np.array:
        """Generate random parameters
        Args:
            nqubits (int): the number of qubits in the circuit
            nlayers (int): the number of layers of ansatz
        Returns:
            array[float]: array of parameters
        """
        if self.initialization_method == "Small":
            params = np.random.uniform(0, np.pi/nqubits/nlayers, size=2 * nqubits * nlayers, requires_grad=True)
        elif self.initialization_method == "MBL":
            constant_params_X = np.random.uniform(0, 0.1, size=nlayers, requires_grad=True)
            random_params_Z   = np.random.uniform(- np.pi, np.pi, size=nqubits * nlayers, requires_grad=True)
            params = np.concatenate([constant_params_X, random_params_Z])
        elif self.initialization_method == "Random":
            params = np.random.uniform(0, 2 * np.pi, size=2 * nqubits * nlayers, requires_grad=True)
        elif self.initialization_method == "Gaussian":
            params = np.random.normal(0, np.pi/obs_locality/nlayers, size=2 * nqubits * nlayers, requires_grad=True)
        else:
            pass

        return params
    
    @staticmethod
    def make_global_observable(nqubits) -> qml.Hermitian:
        """Generate the global observable H = |0><0|^{\otimes n}"""
        H = np.zeros((2**nqubits, 2**nqubits))
        H[0, 0] = 1
        wire_list = [i for i in range(nqubits)]
        
        return qml.Hermitian(H, wire_list)

    def ansatz(self, params, nqubits, nlayers) -> None:
        """Generate the variational quantum circuit.
        Args:
            params (array[float]): array of parameters
            nqubits (int): the number of qubits in the circuit
            nlayers (int): the number of layers of ansatz
        Returns:
            expectation value (float): the expectation value of the target observable
        """
        if self.initialization_method in ("Small", "Random", "Gaussian"):
            for i in range(nlayers):
                for j in range(nqubits):
                    qml.RX(params[nqubits * i + j], wires=j)
                    qml.RZ(params[nqubits * (nlayers + i) + j], wires=j)
                
                for j in range(0, nqubits-1, 2):
                    qml.CZ(wires=[j, j+1])
                for j in range(1, nqubits-1, 2):
                    qml.CZ(wires=[j, j+1])
                qml.Barrier()
        elif self.initialization_method == "MBL":
            for i in range(nlayers):
                for j in range(nqubits):
                    qml.RX(params[i], wires=j)
                    qml.RZ(params[nlayers + nqubits * i + j], wires=j)
                
                for j in range(0, nqubits-1, 2):
                    qml.CZ(wires=[j, j+1])
                for j in range(1, nqubits-1, 2):
                    qml.CZ(wires=[j, j+1])
                qml.Barrier()
        else:
            pass

    def make_circuit(self, nqubits) -> qml.QNode:
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
            
            if self.observable is None:
                self.obs_locality = nqubits
            else:
                self.obs_locality = len(list(self.observable.wires))

            for nlayers in self.nlayers_list:
                gradients_list = []
                qcircuit = self.make_circuit(nqubits)
                grad = qml.grad(qcircuit, argnum=0)

                for _ in range(self.nsamples):
                    params = self.make_initial_params(nqubits, nlayers, self.obs_locality)
                    gradients = grad(params, nqubits, nlayers)
                    gradients_list.append(gradients)

                # By transpose, each list in gradients_list have nsample gradients of one parameter.
                gradients_list = np.array(gradients_list).T

                grad_means.append(
                    np.mean(
                        [np.mean(row) for row in gradients_list]
                    )
                )
                grad_vars.append(
                    np.mean(
                        # [np.mean(row**2) for row in gradients_list]
                        [np.var(row) for row in gradients_list]
                    )
                )

            self.grad_means_list.append(grad_means)
            self.grad_vars_list.append(grad_vars)
            print(f"nqubits {nqubits},  observable size 2^{self.obs_locality} * 2^{self.obs_locality}")

        # return self.grad_means_list, self.grad_vars_list

    def plot_bp_nlayers(self, x_axis='nqubits', save=False, mean=False):
        
        if x_axis in ('nlayers', 'nqubits'):
            pass
        else:
            raise ValueError("x_axis must be 'nlayers' or 'nqubits'")
        
        if mean:
            if x_axis == 'nlayers':
                for i, nqubits in enumerate(self.nqubits_list):
                    plt.plot(self.nlayers_list, self.grad_means_list[i], label=f"nqubits = {nqubits}", marker="o", linestyle='dashed')

                plt.plot(self.nlayers_list, np.zeros(len(self.nlayers_list)), color="black", linestyle='dashed')
                plt.xlabel(r"L", fontsize=18)
                plt.ylabel(r"$E[ \partial_{\theta_i} C ]$", fontsize=18)
                plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=18)
                plt.title(f"HEA, mean, {self.initialization_method}, {(self.observable is None)*'Global'+(self.observable is not None)*'Local'} Obs", fontsize=18)
                plt.show()
            else:
                for j, nlayers in enumerate(self.nlayers_list):
                    plt.plot(self.nqubits_list, np.array(self.grad_means_list).T[j], label=f"L = {nlayers}", marker="o", linestyle='dashed')

                plt.plot(self.nqubits_list, np.zeros(len(self.nqubits_list)), color="black", linestyle='dashed')
                plt.xlabel(r"N Qubits", fontsize=18)
                plt.ylabel(r"$E[ \partial_{\theta_i} C ]$", fontsize=18)
                plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=18)
                plt.title(f"HEA, mean, {self.initialization_method}, {(self.observable is None)*'Global'+(self.observable is not None)*'Local'} Obs", fontsize=18)
                plt.show()
        else:
            pass
        
        if x_axis == 'nlayers':
            for i, nqubits in enumerate(self.nqubits_list):
                plt.semilogy(self.nlayers_list, self.grad_vars_list[i], label=f"n = {nqubits}", marker="o", linestyle='dashed')
            plt.xlabel(r"L", fontsize=18)
            plt.ylabel("$\mathrm{Var}[\partial_{\\theta} C]$", fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.ylim([1e-7, 1e-0])
            plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=18)
            plt.title(f"HEA, variance, {self.initialization_method}, {(self.observable is None)*'Global'+(self.observable is not None)*'Local'} Obs", fontsize=18)
            if save:
                plt.savefig(f"HEA-{self.initialization_method}-{(self.observable is None)*'global'+(self.observable is not None)*'local'}-ob.pdf", bbox_inches="tight")
            else:
                pass
            plt.show()
        else:
            for j, nlayers in enumerate(self.nlayers_list):
                plt.semilogy(self.nqubits_list, np.array(self.grad_vars_list).T[j], label=f"L = {nlayers}", marker="o", linestyle='dashed')
            plt.xlabel(r"N Qubits", fontsize=18)
            plt.ylabel("$\mathrm{Var}[\partial_{\\theta} C]$", fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.ylim([1e-7, 1e-0])
            plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=18)
            plt.title(f"HEA, variance, {self.initialization_method}, {(self.observable is None)*'Global'+(self.observable is not None)*'Local'} Obs", fontsize=18)
            if save:
                plt.savefig(f"HEA-{self.initialization_method}-{(self.observable is None)*'global'+(self.observable is not None)*'local'}-ob.pdf", bbox_inches="tight")
            else:
                pass
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
            plt.title(f"HEA, mean, {self.initialization_method}, {(self.observable is None)*'Global'+(self.observable is not None)*'Local'} Obs", fontsize=18)
            plt.show()
        else:
            pass

        # Fit the semilog plot to a straight line
        p = np.polyfit(
            x=self.nqubits_list,
            y=np.log(np.clip(a=np.ravel(self.grad_vars_list), a_min=1e-323, a_max=None)),
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
        plt.ylabel("$\mathrm{Var}[\partial_{\\theta} C]$", fontsize=18)
        plt.legend()
        plt.title(f"HEA, variance, {self.initialization_method}, {(self.observable is None)*'Global'+(self.observable is not None)*'Local'} Obs", fontsize=18)
        plt.show()
