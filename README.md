## QR-SDN: Reinforcement Learning for Direct Flow Routing in Software-Defined Networks


#### Folder description
- `/controller`:
Includes the controller implementation.
Scripts can be started with: `ryu-manager remote_controller_RL.py`
    - `controller/config.py`:
        Config file containing all the parameters for the learning agent/controller (e.g. exploration strategy)

- `/mininet`:
Includes the Mininet emulation scripts with different topologies.
Scripts can be started with: `sudo python Four_switches_two_ways.py`

- `/plots`:
Includes the measurement data and plot scripts.
Scripts can be started with: `python3 plot_<scenario>.py`


#### TODO's

- add Jupyter notebook to produce plots
- improve documentation