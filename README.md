
# NeuronGPU
A GPU-MPI library for simulation of large-scale networks of spiking neurons.
Can be used in Python, in C++ and in C.

With this library it is possible to run relatively fast simulations of large-scale networks of spiking neurons. For instance, on a single Nvidia GeForce RTX 2080 Ti GPU board it is possible to simulate the activity of 1 million multisynapse AdEx neurons with 1000 synapse per neurons, for a total of 1 billion synapse, in little more than 100 seconds per second of neural activity.
The MPI communication is also very efficient.
The python interface is very similar to that of the NEST simulator: the most used commands are practically identical, dictionaries are used to define neurons, connections and synapsis properties in the same way.

To start using it,, have a look at the examples in the python/examples and c++/examples folders.

* **[Download and installation instructions](https://github.com/golosio/NeuronGPU/wiki/Installation-instructions)**
* **[User guide](https://github.com/golosio/NeuronGPU/wiki)**
