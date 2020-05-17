import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import linalg, pi as pi, sin as sin, cos as cos, arctan as atan, arccos as acos, exp as exp
from ReceptorModel import Receptor
from NanoparticleModel import Nanoparticle
from LigandModel import Ligand
import pandas as pd
from numba import njit
from random import shuffle


class MyModel:
    def __init__(self, dimension, binding_energy, time_unit, number_of_receptors, receptor_length, number_of_nanoparticles,
                 nanoparticle_radius, number_of_ligands, ligand_length, binding_distance):
        self.agents = []  # create an empty list of agents
        self.collision = False
        self.count = 0
        self.dimension = dimension
        self.points = []
        self.binding_energy = binding_energy
        self.time_unit = time_unit
        self.number_of_receptors = number_of_receptors
        self.receptor_length = receptor_length
        self.number_of_nanoparticles = number_of_nanoparticles
        self.number_of_ligands = number_of_ligands
        self.nanoparticle_radius = nanoparticle_radius
        self.ligand_length = ligand_length
        self.binding_distance = binding_distance

    def create_nanoparticles_and_ligands(self):  # Adds nanoparticles into the system
        true_radius = self.nanoparticle_radius + self.ligand_length  # The maximum radius of the nanoparticle including the ligands
        upper_limit = self.dimension - true_radius
        for i in range(self.number_of_nanoparticles):  # loop from 0 to number of agents
            agent_id = f'Nanoparticle {i}'
            while True:
                nanoparticle_position_xyz = np.array(
                    [np.random.uniform(true_radius, upper_limit), np.random.uniform(true_radius, upper_limit),
                     np.random.uniform(true_radius, upper_limit)])  # 3D cube cartesian system - rectangular coordinates
                if self.is_space_available_nanoparticle(nanoparticle_position_xyz):
                    break
                else:
                    continue
            nanoparticle = Nanoparticle(agent_id, nanoparticle_position_xyz, self.number_of_ligands, self.nanoparticle_radius,
                                        self.ligand_length, self.dimension, binding_energy=self.binding_energy,
                                        time_unit=self.time_unit)
            # print(f'Created {agent_id}')
            self.agents.append(nanoparticle)  # add agent to list of agents

    def is_space_available_nanoparticle(self, attempt):  # makes sure new nanoparticles don't overlap existing ones
        count = 0
        nanoparticle_positions = [agent.position for agent in self.agents if isinstance(agent, Nanoparticle)]
        separation = (self.nanoparticle_radius + self.ligand_length) * 2
        for i in nanoparticle_positions:
            distance = self.distance(attempt, i)
            if distance >= separation:  # Checks that if the nanoparticle is a certain distance from the other nanoparticles
                count += 1
            else:
                return False
        if count == len(nanoparticle_positions):
            receptor_positions = [agent.position for agent in self.agents if isinstance(agent, Receptor)]
            if len(receptor_positions) > 0:
                separation2 = 7  # receptor must be at least a radius away from receptor
                count = 0
                for i in receptor_positions:
                    distance = self.distance(attempt, i)
                    if distance >= separation2:  # Checks that if the nanoparticle is a certain distance from the other nanoparticles
                        count += 1
                    else:
                        return False  # Repulsive potential not exceede
                if count == len(receptor_positions):
                    return True  # Returns True when there is space available
            else:
                return True  # if no receptors in the system

    def create_receptors(self):  # Adds receptors into the system
        for i in range(self.number_of_receptors):  # loop from 0 to number of agents
            receptor_id = f'Receptor {i}'
            base_position = np.array([np.random.uniform(0, self.dimension), np.random.uniform(0, self.dimension), 0])  # 3D cube cartesian system - rectangular coordinates
            receptor = Receptor(receptor_id, base_position, self.receptor_length, self.dimension, self.binding_energy, self.nanoparticle_radius, self.ligand_length)
            self.agents.append(receptor)  # add agent to list of agents

    def is_space_available_receptor(self, attempt):  # makes sure new nanoparticles don't overlap existing ones
        count = 0
        receptor_positions = [agent.base_position for agent in self.agents if isinstance(agent, Receptor)]
        separation = self.number_of_receptors * 0.025
        for i in receptor_positions:
            distance = self.distance(attempt, i)
            if distance >= separation:  # Checks that if the receptor is a certain distance from the other nanoparticles
                count += 1
            else:
                return False
        if count == len(receptor_positions):
            nanoparticle_positions = [agent.position for agent in self.agents if isinstance(agent, Nanoparticle)]
            if len(nanoparticle_positions) > 0:
                separation2 = self.nanoparticle_radius + self.ligand_length  # nanoparticle must be at least a radius away from receptor
                count = 0
                for i in nanoparticle_positions:
                    distance = self.distance(attempt, i)
                    if distance >= separation2:  # Checks that if the nanoparticle is a certain distance from the other nanoparticles
                        count += 1
                    else:
                        return False  # Repulsive potential not exceede
                if count == len(nanoparticle_positions):
                    return True  # Returns True when there is space available
            else:
                return True  # if no nanoparticles in the system

    def run(self, steps):
        # self.steric_potential = self.steric_potential()
        self.coverage = [0]
        self.time = 0
        plateau_count = 0
        self.bound_nanoparticles = 0
        for i in range(steps):  # loop through number of steps
            self.time += 1
            # print(f'Running step {i} ---------------------------------------------------------------------------------')
            self.step()  # run one step
            bound_nanoparticles = 0
            for agent in self.agents:
                if isinstance(agent, Nanoparticle):
                    if agent.bound is True:
                        bound_nanoparticles += 1  # Number of nanoparticles bound to the cell surface
            self.coverage.append(self.calculate_surface_coverage(bound_nanoparticles))
            '''if you wish experiment to stop at a plateau + visually see the interaction'''
            # if i == steps - 1:  # or i == steps/2:
            #     self.visualiser()
            # self.visualiser()
            # if self.time > 200:  # change accordingly
            #     if len(self.coverage) > 2:
            #         if (self.coverage[-1] - self.coverage[-2]) < 0.005:
            #             plateau_count += 1
            #     if plateau_count > 50:
            #         self.visualiser()
            #         break  # If data starts to plateau for 50 steps, then the experiment ends
        for agent in self.agents:
            if isinstance(agent, Nanoparticle):
                if agent.bound is True:
                    self.bound_nanoparticles += 1  # Number of nanoparticles bound to the cell surface
        self.calculate_surface_coverage(self.bound_nanoparticles)
        count1 = 0
        count2 = 0
        for i in self.agents:
            if isinstance(i, Nanoparticle) and i.bound is True:
                count1 += 1
            elif isinstance(i, Receptor) and i.bound is not None:
                count2 += 1
        print(f'There are {count1} nanoparticles bound to the surface')
        print(f'There are {count2} receptors bound to nanoparticles')

    def step(self):
        list_of_nanoparticle_arrays = list(np.random.normal(size=(self.number_of_nanoparticles, 3)))
        list_of_receptor_arrays = list(np.random.normal(size=(self.number_of_receptors, 3)))
        self.nanoparticle_list = [agent.position for agent in self.agents if isinstance(agent, Nanoparticle)]
        self.receptor_list = [agent.position for agent in self.agents if isinstance(agent, Receptor)]
        max_dist_to_react = self.nanoparticle_radius + self.ligand_length + self.receptor_length  # Loop only entered if Nanoparticle is close enough, i.e. receptor have a base position where z = 0
        max_dist_to_react2 = self.nanoparticle_radius + self.ligand_length
        for agent in self.agents:  # loop through agents
            if isinstance(agent, Nanoparticle):
                nanoparticles_list = [nanoparticle_position for nanoparticle_position in self.nanoparticle_list if nanoparticle_position is not agent.position]  # List of the other nanoparticles' positions
                agent.step(self.nanoparticle_brownian(list_of_nanoparticle_arrays.pop()), nanoparticles_list, self.receptor_list)  # different movements for nanoparticle and its ligands
                if agent.position[2] < max_dist_to_react:  # Reaction only possible only if the ligand is close enough to the in the z-axis to the receptor
                    # print(f'{agent.agent_id} is interacting -------------------------------------------------------------------------------------------')
                    receptors = [x for x in self.agents if isinstance(x, Receptor) and x.bound is None]  # List of receptors that are not bound to a ligand
                    for r in receptors:
                        # print(f'{agent.agent_id} is interacting -------------------------------------------------------------------------------------------')
                        # if linalg.norm(agent.position - r.position) < max_dist_to_react2:  # Loop only entered if Nanoparticle is close enough to receptors
                        if max_dist_to_react2 <= self.distance(agent.position, r.position) < max_dist_to_react2 + 1:  # Loop only entered if Nanoparticle is close enough to receptors
                            self.interaction_criteria(agent, r)  # if any agent meets interaction criteria with "agent"
            elif isinstance(agent, Receptor):
                receptors_list = [receptor_position for receptor_position in self.receptor_list if receptor_position is not agent.position]  # List of the other receptors' base positions
                agent.step(self.receptor_brownian(list_of_receptor_arrays.pop()), receptors_list, self.nanoparticle_list)  # move agent
                if agent.bound is None:  # Checks that the receptor isn't already bound to a ligand
                    nanoparticles = [n for n in self.agents if isinstance(n, Nanoparticle)]  # and (n.position[2] < max_dist_to_react))]  # List of Nanoparticles
                    for n in nanoparticles:  # loop through agents again
                        # print(f'{agent.agent_id} is interacting -------------------------------------------------------------------------------------------')
                        # if linalg.norm(agent.position - n.position) < max_dist_to_react2:  # Loop only entered if Nanoparticle is close enough to receptors
                        if self.distance(agent.position, n.position) < max_dist_to_react2:  # Loop only entered if Nanoparticle is close enough to receptors
                            self.interaction_criteria(n, agent)  # if any agent meets interaction criteria with "agent"
            else:
                print(False)

    def calculate_surface_coverage(self, n):
        self.surface_coverage = n * ((4 * (self.nanoparticle_radius + self.ligand_length) ** 2) / self.dimension ** 2)
        return self.surface_coverage

    def visualiser(self):
        """Nanoparticles"""
        self.agent_nanoparticles_dictionary = {'Points': 'Bound'}
        for agent in self.agents:
            if isinstance(agent, Nanoparticle):
                self.agent_nanoparticles_dictionary[tuple(agent.position.tolist())] = agent.bound
        nanoparticles = [agent for agent in self.agents if isinstance(agent, Nanoparticle)]
        x = [i.position[0] for i in nanoparticles]
        y = [i.position[1] for i in nanoparticles]
        z = [i.position[2] for i in nanoparticles]
        bound_nanoparticles = [str(i.bound) for i in nanoparticles]
        dictionary = {'x': x, 'y': y, 'z': z, 'Bound': bound_nanoparticles}
        self.nanoparticles_df = pd.DataFrame(dictionary)
        x1 = []
        y1 = []
        z1 = []
        x2 = []
        y2 = []
        z2 = []
        for position, bound in self.agent_nanoparticles_dictionary.items():
            if bound is False:
                x1.append(position[0])
                y1.append(position[1])
                z1.append(position[2])
            if bound is True:
                x2.append(position[0])
                y2.append(position[1])
                z2.append(position[2])
        '''Receptors'''
        self.agent_receptors_dictionary = {'Points': 'Bound'}
        for agent in self.agents:
            if isinstance(agent, Receptor):
                self.agent_receptors_dictionary[tuple(agent.position.tolist())] = agent.bound
        receptors = [agent for agent in self.agents if isinstance(agent, Receptor)]
        x3 = [i.position[0] for i in receptors]
        y3 = [i.position[1] for i in receptors]
        z3 = [i.position[2] for i in receptors]
        bound_receptors = [str(i.bound) for i in receptors]
        dictionary = {'x': x3, 'y': y3, 'z': z3, 'Bound': bound_receptors}
        self.receptors_df = pd.DataFrame(dictionary)
        x4 = []
        y4 = []
        z4 = []
        x5 = []
        y5 = []
        z5 = []
        for position, bound in self.agent_receptors_dictionary.items():
            if bound is None:
                x4.append(position[0])
                y4.append(position[1])
                z4.append(position[2])
            elif bound is not (None or 'Bound'):
                x5.append(position[0])
                y5.append(position[1])
                z5.append(position[2])
        '''Ligands'''
        x6 = []
        y6 = []
        z6 = []
        for agent in self.agents:  # loop through agents
            if isinstance(agent, Nanoparticle):
                for i in agent.ligands:
                    x6.append(i.position[0])
                    y6.append(i.position[1])
                    z6.append(i.position[2])
        '''3D'''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x1, y1, z1, c='Red', s=350 * (self.nanoparticle_radius / 50))  # Unbound
        ax.scatter(x2, y2, z2, c='Green', s=350 * (self.nanoparticle_radius / 50))  # Bound
        ax.scatter(x4, y4, z4, c='Red', s=self.receptor_length, marker='1')  # Unbound
        ax.scatter(x5, y5, z5, c='Green', s=self.receptor_length, marker='1')  # Bound
        # ax.scatter(x6, y6, z6, c='Blue', s=self.ligand_length, marker='1')  # Ligands
        ax.set_title('3D')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_xlim(0, self.dimension)
        ax.set_ylim(0, self.dimension)
        ax.set_zlim(0, self.dimension)
        plt.show()
        '''2D'''
        # plt.scatter(x1, y1, c='Red', s=2 * (self.nanoparticle_radius + self.ligand_length))
        # plt.scatter(x2, y2, c='Green', s=2 * (self.nanoparticle_radius + self.ligand_length))
        # plt.title('2D')
        # plt.xlabel('X Axis')
        # plt.ylabel('Y Axis')
        # plt.xlim(0, self.dimension)
        # plt.ylim(0, self.dimension)
        # plt.show()

    def nanoparticle_brownian(self, array):
        random_movement_cartesian = ((2 * ((1.38064852e-23 * 310.15) / (
                6 * pi * 8.9e-4 * (self.nanoparticle_radius * 1e-9)))) ** 0.5) * 1e9 * self.time_unit * array
        '''using viscosity of water'''
        return random_movement_cartesian

    def receptor_brownian(self, array):
        """look for the end of the receptors or 100 times smaller"""
        random_movement_cartesian = ((2 * ((1.38064852e-23 * 310.15) / (
                6 * pi * 8.9e-4 * (self.nanoparticle_radius * 1e-9 / 100)))) ** 0.5) * 1e9 * self.time_unit * array
        '''using viscosity of water'''
        r = (random_movement_cartesian[0] ** 2 + random_movement_cartesian[1] ** 2 + random_movement_cartesian[
            2] ** 2) ** 0.5
        θ = atan(random_movement_cartesian[1] / random_movement_cartesian[0])
        Φ = acos(random_movement_cartesian[2] / r)
        '''Equations from: https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Calculus_(OpenStax)/12%3A_Vectors_in_Space/12.7%3A_Cylindrical_and_Spherical_Coordinates'''
        '''simply explained in https://mathworld.wolfram.com/SphericalCoordinates.html'''
        return np.array([r, θ, Φ])

    def interaction_criteria(self, nanoparticle, receptor):
        if nanoparticle.agent_id != receptor.agent_id:  # true if same id
            ligands = nanoparticle.ligands
            for ligand in ligands:
                if ligand.bound is None and receptor.bound is None:  # Can only bind one receptor to one ligand at a time
                    distance = self.distance(ligand.position, receptor.position)
                    if distance <= self.binding_distance:  # true if close position, false otherwise
                        # print(f'Collision with {ligand.agent_id} of {agent1.agent_id} and {agent2.agent_id}')
                        if np.random.uniform(low=0, high=1) > exp(-self.binding_energy):
                            # print(f'Reaction happened between {ligand.agent_id} of {nanoparticle.agent_id} and {receptor.agent_id}')
                            ligand.bound = receptor
                            receptor.bound = ligand
                            nanoparticle.bound = True
                            ligand.position = receptor.position
                            self.count += 1
                            break  # As receptor can only bind to one ligand
                        else:
                            continue

    @staticmethod
    @njit(fastmath=True)
    def distance(a, b):
        return linalg.norm(a - b)

    
def second_variable_plot(x, y, list1, list2, errors):
    plt.xlabel(x)
    plt.ylabel(y)
    list1.insert(0, 0)
    list2.insert(0, 0)
    errors.insert(0, 0)
    plt.errorbar(list1, list2, yerr=errors)
    plt.show()


def binding_energy():
    print('Binding Energy -------------')
    d = np.linspace(5, 25, 5).tolist()
    data = {}
    means = []
    errors = []
    time_data = []
    for i in d:
        variable_finals = []
        for j in range(3):
            number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
            my_model = MyModel(dimension=1000, binding_energy=int(i), time_unit=10e-3, number_of_receptors=1000, receptor_length=100,
                               number_of_nanoparticles=190, nanoparticle_radius=50, number_of_ligands=100, ligand_length=7, binding_distance=4)
            my_model.create_receptors()  # 100 nm for receptor
            my_model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
            print(f'{my_model.dimension/1000} μm\u00b3 system, {my_model.binding_energy} binding energy, {my_model.number_of_nanoparticles} Nanoparticles,\n'
                  f'{my_model.nanoparticle_radius} nm Nanoparticle Radius, {my_model.number_of_ligands} Ligands, Ligand length {my_model.ligand_length} nm,\n'
                  f'{my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length, {my_model.binding_distance} Binding distance')
            my_model.run(steps=number_of_seconds)  # 3600 for 1 hour
            print(f'There were {my_model.count} reactions')
            print(f'The surface coverage is {my_model.surface_coverage}')
            variable_finals.append(my_model.surface_coverage)
            time_data.append(np.array(my_model.coverage))
        mean_time = np.mean(time_data, axis=0)
        error_time = np.std(time_data, axis=0)
        data[f'{my_model.binding_energy} KT binding energy '] = np.array([list(range(0, my_model.time + 1)), mean_time, error_time])
        mean_coverage = np.mean(np.array(variable_finals))
        print(f'The mean surface coverage is {mean_coverage}')
        errors.append(np.std(np.array(variable_finals)))
        means.append(np.mean(np.array(variable_finals)))
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Surface Coverage')
    for key, value in data.items():
        plt.plot(value[0], value[1], label=key)
        plt.fill_between(value[0], value[1] - value[2], value[1] + value[2], alpha=0.2)
    plt.legend()
    plt.show()
    second_variable_plot('Binding energy (kt)', 'Surface Coverage', d, means, errors)


binding_energy()


def number_of_receptors():
    print('Number of receptors -------------')
    d = np.linspace(250, 1000, 4).tolist()
    data = {}
    means = []
    errors = []
    time_data = []
    for i in d:
        variable_finals = []
        for j in range(3):
            number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
            my_model = MyModel(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=int(i),
                               receptor_length=100,
                               number_of_nanoparticles=190, nanoparticle_radius=50, number_of_ligands=100,
                               ligand_length=7, binding_distance=4)
            my_model.create_receptors()  # 100 nm for receptor
            my_model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
            print(f'{my_model.dimension / 1000} μm\u00b3 system, {my_model.binding_energy} binding energy, {my_model.number_of_nanoparticles} Nanoparticles,\n'
                  f'{my_model.nanoparticle_radius} nm Nanoparticle Radius, {my_model.number_of_ligands} Ligands, Ligand length {my_model.ligand_length} nm,\n'
                  f'{my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length, {my_model.binding_distance} Binding distance')
            my_model.run(steps=number_of_seconds)  # 3600 for 1 hour
            print(f'There were {my_model.count} reactions')
            print(f'The surface coverage is {my_model.surface_coverage}')
            variable_finals.append(my_model.surface_coverage)
            time_data.append(np.array(my_model.coverage))
        mean_time = np.mean(time_data, axis=0)
        error_time = np.std(time_data, axis=0)
        data[f'{my_model.number_of_receptors} receptors'] = np.array([list(range(0, my_model.time + 1)), mean_time, error_time])
        mean_coverage = np.mean(np.array(variable_finals))
        print(f'The mean surface coverage is {mean_coverage}')
        errors.append(np.std(np.array(variable_finals)))
        means.append(np.mean(np.array(variable_finals)))
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Surface Coverage')
    for key, value in data.items():
        plt.plot(value[0], value[1], label=key)
        plt.fill_between(value[0], value[1] - value[2], value[1] + value[2], alpha=0.2)
    plt.legend()
    plt.show()
    second_variable_plot('Number of receptors', 'Surface Coverage', d, means, errors)


number_of_receptors()


def receptor_length():
    print('Receptor length -------------')
    d = np.linspace(25, 100, 4).tolist()
    data = {}
    means = []
    errors = []
    time_data = []
    for i in d:
        variable_finals = []
        for j in range(3):
            number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
            my_model = MyModel(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=1000,
                               receptor_length=int(i),
                               number_of_nanoparticles=190, nanoparticle_radius=50, number_of_ligands=100,
                               ligand_length=7, binding_distance=4)
            my_model.create_receptors()  # 100 nm for receptor
            my_model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
            print(
                f'{my_model.dimension / 1000} μm\u00b3 system, {my_model.binding_energy} binding energy, {my_model.number_of_nanoparticles} Nanoparticles,\n'
                f'{my_model.nanoparticle_radius} nm Nanoparticle Radius, {my_model.number_of_ligands} Ligands, Ligand length {my_model.ligand_length} nm,\n'
                f'{my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length, {my_model.binding_distance} Binding distance')
            my_model.run(steps=number_of_seconds)  # 3600 for 1 hour
            print(f'There were {my_model.count} reactions')
            print(f'The surface coverage is {my_model.surface_coverage}')
            variable_finals.append(my_model.surface_coverage)
            time_data.append(np.array(my_model.coverage))
        mean_time = np.mean(time_data, axis=0)
        error_time = np.std(time_data, axis=0)
        data[f'{my_model.receptor_length} nm receptor length'] = np.array(
            [list(range(0, my_model.time + 1)), mean_time, error_time])
        mean_coverage = np.mean(np.array(variable_finals))
        print(f'The mean surface coverage is {mean_coverage}')
        errors.append(np.std(np.array(variable_finals)))
        means.append(np.mean(np.array(variable_finals)))
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Surface Coverage')
    for key, value in data.items():
        plt.plot(value[0], value[1], label=key)
        plt.fill_between(value[0], value[1] - value[2], value[1] + value[2], alpha=0.2)
    plt.legend()
    plt.show()
    second_variable_plot('Receptor length (nm)', 'Surface Coverage', d, means, errors)


receptor_length()


def number_of_nanoparticles():
    print('Number of nanoparticles -------------')
    d = np.linspace(60, 180, 3).tolist()
    d.append(190)
    data = {}
    means = []
    errors = []
    time_data = []
    for i in d:
        variable_finals = []
        for j in range(3):
            number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
            my_model = MyModel(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=1000,
                               receptor_length=100,
                               number_of_nanoparticles=int(i), nanoparticle_radius=50, number_of_ligands=100,
                               ligand_length=7, binding_distance=4)
            my_model.create_receptors()  # 100 nm for receptor
            my_model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
            print(
                f'{my_model.dimension / 1000} μm\u00b3 system, {my_model.binding_energy} binding energy, {my_model.number_of_nanoparticles} Nanoparticles,\n'
                f'{my_model.nanoparticle_radius} nm Nanoparticle Radius, {my_model.number_of_ligands} Ligands, Ligand length {my_model.ligand_length} nm,\n'
                f'{my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length, {my_model.binding_distance} Binding distance')
            my_model.run(steps=number_of_seconds)  # 3600 for 1 hour
            print(f'There were {my_model.count} reactions')
            print(f'The surface coverage is {my_model.surface_coverage}')
            variable_finals.append(my_model.surface_coverage)
            time_data.append(np.array(my_model.coverage))
        mean_time = np.mean(time_data, axis=0)
        error_time = np.std(time_data, axis=0)
        data[f'{my_model.number_of_nanoparticles} nanoparticles'] = np.array(
            [list(range(0, my_model.time + 1)), mean_time, error_time])
        mean_coverage = np.mean(np.array(variable_finals))
        print(f'The mean surface coverage is {mean_coverage}')
        errors.append(np.std(np.array(variable_finals)))
        means.append(np.mean(np.array(variable_finals)))
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Surface Coverage')
    for key, value in data.items():
        plt.plot(value[0], value[1], label=key)
        plt.fill_between(value[0], value[1] - value[2], value[1] + value[2], alpha=0.2)
    plt.legend()
    plt.show()
    second_variable_plot('Number of nanoparticles', 'Surface Coverage', d, means, errors)


number_of_nanoparticles()


def nanoparticle_radius():
    print('Nanoparticle Radius -------------')
    d = np.linspace(10, 50, 5).tolist()
    data = {}
    means = []
    errors = []
    time_data = []
    for i in d:
        variable_finals = []
        for j in range(3):
            number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
            my_model = MyModel(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=1000,
                               receptor_length=100,
                               number_of_nanoparticles=190, nanoparticle_radius=int(i), number_of_ligands=100,
                               ligand_length=7, binding_distance=4)
            my_model.create_receptors()  # 100 nm for receptor
            my_model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
            print(
                f'{my_model.dimension / 1000} μm\u00b3 system, {my_model.binding_energy} binding energy, {my_model.number_of_nanoparticles} Nanoparticles,\n'
                f'{my_model.nanoparticle_radius} nm Nanoparticle Radius, {my_model.number_of_ligands} Ligands, Ligand length {my_model.ligand_length} nm,\n'
                f'{my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length, {my_model.binding_distance} Binding distance')
            my_model.run(steps=number_of_seconds)  # 3600 for 1 hour
            print(f'There were {my_model.count} reactions')
            print(f'The surface coverage is {my_model.surface_coverage}')
            variable_finals.append(my_model.surface_coverage)
            time_data.append(np.array(my_model.coverage))
        mean_time = np.mean(time_data, axis=0)
        error_time = np.std(time_data, axis=0)
        data[f'{my_model.nanoparticle_radius} nm nanoparticle radius'] = np.array(
            [list(range(0, my_model.time + 1)), mean_time, error_time])
        mean_coverage = np.mean(np.array(variable_finals))
        print(f'The mean surface coverage is {mean_coverage}')
        errors.append(np.std(np.array(variable_finals)))
        means.append(np.mean(np.array(variable_finals)))
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Surface Coverage')
    for key, value in data.items():
        plt.plot(value[0], value[1], label=key)
        plt.fill_between(value[0], value[1] - value[2], value[1] + value[2], alpha=0.2)
    plt.legend()
    plt.show()
    second_variable_plot('Nanoparticle radius (nm)', 'Surface Coverage', d, means, errors)


nanoparticle_radius()


def number_of_ligands():
    print('Number of Ligands -------------')
    d = np.linspace(25, 100, 4).tolist()
    data = {}
    means = []
    errors = []
    time_data = []
    for i in d:
        variable_finals = []
        for j in range(3):
            number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
            my_model = MyModel(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=1000,
                               receptor_length=100,
                               number_of_nanoparticles=190, nanoparticle_radius=50, number_of_ligands=int(i),
                               ligand_length=7, binding_distance=4)
            my_model.create_receptors()  # 100 nm for receptor
            my_model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
            print(
                f'{my_model.dimension / 1000} μm\u00b3 system, {my_model.binding_energy} binding energy, {my_model.number_of_nanoparticles} Nanoparticles,\n'
                f'{my_model.nanoparticle_radius} nm Nanoparticle Radius, {my_model.number_of_ligands} Ligands, Ligand length {my_model.ligand_length} nm,\n'
                f'{my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length, {my_model.binding_distance} Binding distance')
            my_model.run(steps=number_of_seconds)  # 3600 for 1 hour
            print(f'There were {my_model.count} reactions')
            print(f'The surface coverage is {my_model.surface_coverage}')
            variable_finals.append(my_model.surface_coverage)
            time_data.append(np.array(my_model.coverage))
        mean_time = np.mean(time_data, axis=0)
        error_time = np.std(time_data, axis=0)
        data[f'{my_model.number_of_ligands} ligands'] = np.array(
            [list(range(0, my_model.time + 1)), mean_time, error_time])
        mean_coverage = np.mean(np.array(variable_finals))
        print(f'The mean surface coverage is {mean_coverage}')
        errors.append(np.std(np.array(variable_finals)))
        means.append(np.mean(np.array(variable_finals)))
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Surface Coverage')
    for key, value in data.items():
        plt.plot(value[0], value[1], label=key)
        plt.fill_between(value[0], value[1] - value[2], value[1] + value[2], alpha=0.2)
    plt.legend()
    plt.show()
    second_variable_plot('Number of ligands', 'Surface Coverage', d, means, errors)


number_of_ligands()


def ligand_length():
    print('Ligand Length -------------')
    d = np.linspace(1, 7, 4).tolist()
    data = {}
    means = []
    errors = []
    time_data = []
    for i in d:
        variable_finals = []
        for j in range(3):
            number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
            my_model = MyModel(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=1000,
                               receptor_length=100,
                               number_of_nanoparticles=190, nanoparticle_radius=50, number_of_ligands=100,
                               ligand_length=int(i), binding_distance=4)
            my_model.create_receptors()  # 100 nm for receptor
            my_model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
            print(
                f'{my_model.dimension / 1000} μm\u00b3 system, {my_model.binding_energy} binding energy, {my_model.number_of_nanoparticles} Nanoparticles,\n'
                f'{my_model.nanoparticle_radius} nm Nanoparticle Radius, {my_model.number_of_ligands} Ligands, Ligand length {my_model.ligand_length} nm,\n'
                f'{my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length, {my_model.binding_distance} Binding distance')
            my_model.run(steps=number_of_seconds)  # 3600 for 1 hour
            print(f'There were {my_model.count} reactions')
            print(f'The surface coverage is {my_model.surface_coverage}')
            variable_finals.append(my_model.surface_coverage)
            time_data.append(np.array(my_model.coverage))
        mean_time = np.mean(time_data, axis=0)
        error_time = np.std(time_data, axis=0)
        data[f'{my_model.ligand_length} nm ligand length'] = np.array(
            [list(range(0, my_model.time + 1)), mean_time, error_time])
        mean_coverage = np.mean(np.array(variable_finals))
        print(f'The mean surface coverage is {mean_coverage}')
        errors.append(np.std(np.array(variable_finals)))
        means.append(np.mean(np.array(variable_finals)))
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Surface Coverage')
    for key, value in data.items():
        plt.plot(value[0], value[1], label=key)
        plt.fill_between(value[0], value[1] - value[2], value[1] + value[2], alpha=0.2)
    plt.legend()
    plt.show()
    second_variable_plot('Ligand length (nm)', 'Surface Coverage', d, means, errors)


ligand_length()


def binding_distance():
    print('Binding Distance -------------')
    d = np.linspace(2, 6, 3).tolist()
    data = {}
    means = []
    errors = []
    time_data = []
    for i in d:
        variable_finals = []
        for j in range(3):
            number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
            my_model = MyModel(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=1000,
                               receptor_length=100,
                               number_of_nanoparticles=190, nanoparticle_radius=50, number_of_ligands=100,
                               ligand_length=7, binding_distance=int(i))
            my_model.create_receptors()  # 100 nm for receptor
            my_model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
            print(
                f'{my_model.dimension / 1000} μm\u00b3 system, {my_model.binding_energy} binding energy, {my_model.number_of_nanoparticles} Nanoparticles,\n'
                f'{my_model.nanoparticle_radius} nm Nanoparticle Radius, {my_model.number_of_ligands} Ligands, Ligand length {my_model.ligand_length} nm,\n'
                f'{my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length, {my_model.binding_distance} Binding distance')
            my_model.run(steps=number_of_seconds)  # 3600 for 1 hour
            print(f'There were {my_model.count} reactions')
            print(f'The surface coverage is {my_model.surface_coverage}')
            variable_finals.append(my_model.surface_coverage)
            time_data.append(np.array(my_model.coverage))
        mean_time = np.mean(time_data, axis=0)
        error_time = np.std(time_data, axis=0)
        data[f'{my_model.binding_distance} nm binding distance'] = np.array(
            [list(range(0, my_model.time + 1)), mean_time, error_time])
        mean_coverage = np.mean(np.array(variable_finals))
        print(f'The mean surface coverage is {mean_coverage}')
        errors.append(np.std(np.array(variable_finals)))
        means.append(np.mean(np.array(variable_finals)))
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Surface Coverage')
    for key, value in data.items():
        plt.plot(value[0], value[1], label=key)
        plt.fill_between(value[0], value[1] - value[2], value[1] + value[2], alpha=0.2)
    plt.legend()
    plt.show()
    second_variable_plot('Binding distance (nm)', 'Surface Coverage', d, means, errors)


# binding_distance()


'''Code not used'''
    # @staticmethod
    # def repulsive_potential(distance, max_closeness):
    #     repulsive_potential = 4 * 1 * ((max_closeness / distance) ** 12 - (max_closeness / distance) ** 6)
    #     return repulsive_potential
    #
    # def steric_potential(self):
    #     receptor_tip_volume = (4 / 3) * pi * ((self.nanoparticle_radius * 1e-9 / 100) ** 3)
    #     area_per_ligand = (4 * pi * ((self.nanoparticle_radius * 1e-9) ** 2)) / self.number_of_ligands
    #     β = (1.38064852e-23 * 310.15) ** -1  # Boltzmann constant x Temperature (of human body 37°C = 310.15K)
    #     steric_potential = ((receptor_tip_volume * (
    #             (1 - (((self.ligand_length * 1e-9) / (self.receptor_length * 1e-9)) ** 2)) ** (9 / 4)))
    #                         / (area_per_ligand ** (3 / 2))) / β
    #     return steric_potential



'''Repulsive Potential data plot'''
# def repulsive_potential(distance, max_closeness):
#     potential = 4 * 10000 * (((max_closeness / distance) ** 12) - ((max_closeness / distance) ** 6))
#     return potential
#
# nanoparticle_radius = 52  # Radius + ligand
# separation = nanoparticle_radius * 2  # = 104 # separation between two Nanoparticles = Nanoparticle diameter including ligand = 104
# max_closeness_ = separation + 0.5 * nanoparticle_radius  # = 130  # Arbitrary value representing van der Waals' radius
#
# a = []
# # b = [140, 130, 120, 110, 100]
# b = np.arange(100, 500, 1).tolist()
#
# for i in b:
#     a.append(repulsive_potential(i, max_closeness_))
#
# plt.plot(b, a)
# plt.xlabel('Distance')
# plt.ylabel('Potential')
# plt.show()


'''Old code without error bars if need to quickly run'''
# def number_of_receptors():
#     print('Number of receptors -------------')
#     d = np.linspace(200, 1000, 5).tolist()
#     data = {}
#     final_surface_coverage = []
#     for i in d:
#         number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
#         my_model = MyModel(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=int(i), receptor_length=100,
#                            number_of_nanoparticles=190, nanoparticle_radius=50, number_of_ligands=100, ligand_length=7, binding_distance=4)
#         my_model.create_receptors()  # 100 nm for receptor
#         my_model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
#         print(f'{my_model.dimension/1000} μm\u00b3 system, {my_model.binding_energy} binding energy, {my_model.number_of_nanoparticles} Nanoparticles,\n'
#               f'{my_model.nanoparticle_radius} nm Nanoparticle Radius, {my_model.number_of_ligands} Ligands, Ligand length {my_model.ligand_length} nm,\n'
#               f'{my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length, {my_model.binding_distance} Binding distance')
#         my_model.run(steps=number_of_seconds)  # 3600 for 1 hour
#         print(f'There were {my_model.count} reactions')
#         print(f'The surface coverage is {my_model.surface_coverage}')
#         final_surface_coverage.append(my_model.surface_coverage)
#         data[f'{my_model.receptor_length} receptors '] = [list(range(0, my_model.time + 1)), my_model.coverage]
#     plt.xlabel('Time (milliseconds)')
#     plt.ylabel('Surface Coverage')
#     for key, value in data.items():
#         plt.plot(value[0], value[1], label=key)
#     plt.legend()
#     plt.show()
#     second_variable_plot('Number of receptors', 'Surface Coverage', d, final_surface_coverage)
#
#
# # number_of_receptors()
#
#
# def receptor_length():
#     print('Receptor length -------------')
#     d = np.linspace(20, 100, 5).tolist()
#     data = {}
#     final_surface_coverage = []
#     for i in d:
#         number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
#         my_model = MyModel(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=1000, receptor_length=int(i),
#                            number_of_nanoparticles=190, nanoparticle_radius=50, number_of_ligands=100, ligand_length=7, binding_distance=4)
#         my_model.create_receptors()  # 100 nm for receptor
#         my_model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
#         print(f'{my_model.dimension/1000} μm\u00b3 system, {my_model.binding_energy} binding energy, {my_model.number_of_nanoparticles} Nanoparticles,\n'
#               f'{my_model.nanoparticle_radius} nm Nanoparticle Radius, {my_model.number_of_ligands} Ligands, Ligand length {my_model.ligand_length} nm,\n'
#               f'{my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length, {my_model.binding_distance} Binding distance')
#         my_model.run(steps=number_of_seconds)  # 3600 for 1 hour
#         print(f'There were {my_model.count} reactions')
#         print(f'The surface coverage is {my_model.surface_coverage}')
#         final_surface_coverage.append(my_model.surface_coverage)
#         data[f'{my_model.receptor_length} nm receptor length'] = [list(range(0, my_model.time + 1)), my_model.coverage]
#     plt.xlabel('Time (milliseconds)')
#     plt.ylabel('Surface Coverage')
#     for key, value in data.items():
#         plt.plot(value[0], value[1], label=key)
#     plt.legend()
#     plt.show()
#     second_variable_plot('Receptor length (nm)', 'Surface Coverage', d, final_surface_coverage)
#
#
# # receptor_length()
#
#
# def number_of_nanoparticles():
#     print('Number of nanoparticles -------------')
#     d = np.linspace(60, 180, 4).tolist()
#     d.append(190)
#     data = {}
#     final_surface_coverage = []
#     for i in d:
#         number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
#         my_model = MyModel(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=1000, receptor_length=100,
#                            number_of_nanoparticles=int(i), nanoparticle_radius=50, number_of_ligands=100, ligand_length=7, binding_distance=4)
#         my_model.create_receptors()  # 100 nm for receptor
#         my_model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
#         print(f'{my_model.dimension/1000} μm\u00b3 system, {my_model.binding_energy} binding energy, {my_model.number_of_nanoparticles} Nanoparticles,\n'
#               f'{my_model.nanoparticle_radius} nm Nanoparticle Radius, {my_model.number_of_ligands} Ligands, Ligand length {my_model.ligand_length} nm,\n'
#               f'{my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length, {my_model.binding_distance} Binding distance')
#         my_model.run(steps=number_of_seconds)  # 3600 for 1 hour
#         print(f'There were {my_model.count} reactions')
#         print(f'The surface coverage is {my_model.surface_coverage}')
#         final_surface_coverage.append(my_model.surface_coverage)
#         data[f'{my_model.number_of_nanoparticles} nanoparticles'] = [list(range(0, my_model.time + 1)), my_model.coverage]
#     plt.xlabel('Time (milliseconds)')
#     plt.ylabel('Surface Coverage')
#     for key, value in data.items():
#         plt.plot(value[0], value[1], label=key)
#     plt.legend()
#     plt.show()
#     second_variable_plot('Number of nanoparticles', 'Surface Coverage', d, final_surface_coverage)
#
#
# # number_of_nanoparticles()
#
#
# def nanoparticle_radius():
#     print('Nanoparticle Radius -------------')
#     d = np.linspace(10, 50, 5).tolist()
#     data = {}
#     final_surface_coverage = []
#     for i in d:
#         number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
#         my_model = MyModel(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=1000, receptor_length=100,
#                            number_of_nanoparticles=190, nanoparticle_radius=int(i), number_of_ligands=100, ligand_length=7, binding_distance=4)
#         my_model.create_receptors()  # 100 nm for receptor
#         my_model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
#         print(f'{my_model.dimension/1000} μm\u00b3 system, {my_model.binding_energy} binding energy, {my_model.number_of_nanoparticles} Nanoparticles,\n'
#               f'{my_model.nanoparticle_radius} nm Nanoparticle Radius, {my_model.number_of_ligands} Ligands, Ligand length {my_model.ligand_length} nm,\n'
#               f'{my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length, {my_model.binding_distance} Binding distance')
#         my_model.run(steps=number_of_seconds)  # 3600 for 1 hour
#         print(f'There were {my_model.count} reactions')
#         print(f'The surface coverage is {my_model.surface_coverage}')
#         final_surface_coverage.append(my_model.surface_coverage)
#         data[f'{my_model.nanoparticle_radius} nm nanoparticle radius'] = [list(range(0, my_model.time + 1)), my_model.coverage]
#     plt.xlabel('Time (milliseconds)')
#     plt.ylabel('Surface Coverage')
#     for key, value in data.items():
#         plt.plot(value[0], value[1], label=key)
#     plt.legend()
#     plt.show()
#     second_variable_plot('Nanoparticle radius (nm)', 'Surface Coverage', d, final_surface_coverage)
#
#
# # nanoparticle_radius()
#
#
# def number_of_ligands():
#     print('Number of Ligands -------------')
#     d = np.linspace(25, 100, 4).tolist()
#     data = {}
#     final_surface_coverage = []
#     for i in d:
#         number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
#         my_model = MyModel(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=1000, receptor_length=100,
#                            number_of_nanoparticles=190, nanoparticle_radius=50, number_of_ligands=int(i), ligand_length=7, binding_distance=4)
#         my_model.create_receptors()  # 100 nm for receptor
#         my_model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
#         print(f'{my_model.dimension/1000} μm\u00b3 system, {my_model.binding_energy} binding energy, {my_model.number_of_nanoparticles} Nanoparticles,\n'
#               f'{my_model.nanoparticle_radius} nm Nanoparticle Radius, {my_model.number_of_ligands} Ligands, Ligand length {my_model.ligand_length} nm,\n'
#               f'{my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length, {my_model.binding_distance} Binding distance')
#         my_model.run(steps=number_of_seconds)  # 3600 for 1 hour
#         print(f'There were {my_model.count} reactions')
#         print(f'The surface coverage is {my_model.surface_coverage}')
#         final_surface_coverage.append(my_model.surface_coverage)
#         data[f'{my_model.number_of_ligands} ligands'] = [list(range(0, my_model.time + 1)), my_model.coverage]
#     plt.xlabel('Time (milliseconds)')
#     plt.ylabel('Surface Coverage')
#     for key, value in data.items():
#         plt.plot(value[0], value[1], label=key)
#     plt.legend()
#     plt.show()
#     second_variable_plot('Number of ligands', 'Surface Coverage', d, final_surface_coverage)
#
#
# # number_of_ligands()
#
#
# def ligand_length():
#     print('Ligand Length -------------')
#     d = np.linspace(1, 7, 4).tolist()
#     data = {}
#     final_surface_coverage = []
#     for i in d:
#         number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
#         my_model = MyModel(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=1000, receptor_length=100,
#                            number_of_nanoparticles=190, nanoparticle_radius=50, number_of_ligands=100, ligand_length=int(i), binding_distance=4)
#         my_model.create_receptors()  # 100 nm for receptor
#         my_model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
#         print(f'{my_model.dimension/1000} μm\u00b3 system, {my_model.binding_energy} binding energy, {my_model.number_of_nanoparticles} Nanoparticles,\n'
#               f'{my_model.nanoparticle_radius} nm Nanoparticle Radius, {my_model.number_of_ligands} Ligands, Ligand length {my_model.ligand_length} nm,\n'
#               f'{my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length, {my_model.binding_distance} Binding distance')
#         my_model.run(steps=number_of_seconds)  # 3600 for 1 hour
#         print(f'There were {my_model.count} reactions')
#         print(f'The surface coverage is {my_model.surface_coverage}')
#         final_surface_coverage.append(my_model.surface_coverage)
#         data[f'{my_model.ligand_length} nm ligand length'] = [list(range(0, my_model.time + 1)), my_model.coverage]
#     plt.xlabel('Time (milliseconds)')
#     plt.ylabel('Surface Coverage')
#     for key, value in data.items():
#         plt.plot(value[0], value[1], label=key)
#     plt.legend()
#     plt.show()
#     second_variable_plot('Ligand length (nm)', 'Surface Coverage', d, final_surface_coverage)
#
#
# # ligand_length()
#
#
# def binding_distance():
#     print('Binding Distance -------------')
#     d = np.linspace(2, 6, 3).tolist()
#     data = {}
#     final_surface_coverage = []
#     for i in d:
#         number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
#         my_model = MyModel(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=1000, receptor_length=100,
#                            number_of_nanoparticles=190, nanoparticle_radius=50, number_of_ligands=100, ligand_length=7, binding_distance=int(i))
#         my_model.create_receptors()  # 100 nm for receptor
#         my_model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
#         print(f'{my_model.dimension/1000} μm\u00b3 system, {my_model.binding_energy} binding energy, {my_model.number_of_nanoparticles} Nanoparticles,\n'
#               f'{my_model.nanoparticle_radius} nm Nanoparticle Radius, {my_model.number_of_ligands} Ligands, Ligand length {my_model.ligand_length} nm,\n'
#               f'{my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length, {my_model.binding_distance} Binding distance')
#         my_model.run(steps=number_of_seconds)  # 3600 for 1 hour
#         print(f'There were {my_model.count} reactions')
#         print(f'The surface coverage is {my_model.surface_coverage}')
#         final_surface_coverage.append(my_model.surface_coverage)
#         data[f'{my_model.binding_distance} nm Binding distance '] = [list(range(0, my_model.time + 1)), my_model.coverage]
#     plt.xlabel('Time (milliseconds)')
#     plt.ylabel('Surface Coverage')
#     for key, value in data.items():
#         plt.plot(value[0], value[1], label=key)
#     plt.legend()
#     plt.show()
#     second_variable_plot('Binding distance (nm)', 'Surface Coverage', d, final_surface_coverage)
#
#
# # binding_distance()
#
#
# def stored_visualiser(dimensions, data):
#     df = pd.read_csv(data)
#     x1 = []
#     y1 = []
#     z1 = []
#     x2 = []
#     y2 = []
#     z2 = []
#     for i in range(len(df.index)):
#         a = str(df.iloc[i]['Bound'])
#         if a == 'True':
#             x1.append(df.iloc[i]['x'])
#             y1.append(df.iloc[i]['y'])
#             z1.append(df.iloc[i]['z'])
#         if a == 'False':
#             x2.append(df.iloc[i]['x'])
#             y2.append(df.iloc[i]['y'])
#             z2.append(df.iloc[i]['z'])
#     '''3D'''
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x1, y1, z1, c='Green', s=104)  # Bound
#     ax.scatter(x2, y2, z2, c='Red', s=104)  # Unbound
#     ax.set_title('3D')
#     ax.set_xlabel('X axis')
#     ax.set_ylabel('Y axis')
#     ax.set_zlabel('Z axis')
#     ax.set_xlim(0, dimensions)
#     ax.set_ylim(0, dimensions)
#     ax.set_zlim(0, dimensions)
#     plt.show()

'''Get graph from data'''
# df = pd.read_csv('data12.CSV', header=0)
# plt.title(f'Number of Ligands = 10^{2}')
# plt.xlabel('Time (Seconds)')
# plt.ylabel('Surface Coverage')
# plt.plot(df['Time (Seconds)'], df['Surface Coverage'])
# plt.show()

'''old code below'''
# def stored_visualiser(dimensions, data):
#     df = pd.read_csv(data)
#     x1 = []
#     y1 = []
#     z1 = []
#     x2 = []
#     y2 = []
#     z2 = []
#     for i in range(len(df.index)):
#         a = str(df.iloc[i]['Bound'])
#         if a == 'True':
#             x1.append(df.iloc[i]['x'])
#             y1.append(df.iloc[i]['y'])
#             z1.append(df.iloc[i]['z'])
#         if a == 'False':
#             x2.append(df.iloc[i]['x'])
#             y2.append(df.iloc[i]['y'])
#             z2.append(df.iloc[i]['z'])
#     '''3D'''
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x1, y1, z1, c='Green', s=104)  # Bound
#     ax.scatter(x2, y2, z2, c='Red', s=104)  # Unbound
#     ax.set_title('3D')
#     ax.set_xlabel('X axis')
#     ax.set_ylabel('Y axis')
#     ax.set_zlabel('Z axis')
#     ax.set_xlim(0, dimensions)
#     ax.set_ylim(0, dimensions)
#     ax.set_zlim(0, dimensions)
#     plt.show()


# stored_visualiser(dimensions=1000, data='data11points.CSV')

# def experiment_time():
#     d = np.linspace(10, 100, 10).tolist()
#     data = {}
#     final_surface_coverage = []
#     for i in d:
#         number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
#         my_model = MyModel(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=1000, receptor_length=int(i),
#                            number_of_nanoparticles=190, number_of_ligands=100, nanoparticle_radius=50, ligand_length=7, binding_distance=4)
#         my_model.create_receptors()  # 100 nm for receptor
#         my_model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
#         print(f'{my_model.dimension/1000} μm\u00b3 system, {my_model.binding_energy} binding energy {my_model.number_of_nanoparticles} Nanoparticles,\n'
#               f'{my_model.nanoparticle_radius} nm Nanoparticle Radius, {my_model.number_of_ligands} Ligands, Ligand length {my_model.ligand_length} nm,\n'
#               f'{my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length, {my_model.binding_distance} Binding distance')
#         my_model.run(steps=number_of_seconds)  # 3600 for 1 hour
#         print(f'There were {my_model.count} reactions')
#         print(f'The surface coverage is {my_model.surface_coverage}')
#         final_surface_coverage.append(my_model.surface_coverage)
#         # points = my_model.agent_position_dictionary
#         '''Plot Time (Seconds) v surface coverage'''
#         # data[f'Binding distance {my_model.binding_distance} nm'] = [list(range(0, my_model.time + 1)), my_model.coverage]
#         # data[f'{my_model.number_of_ligands} ligands'] = [list(range(0, my_model.time + 1)), my_model.coverage]
#         # data[f'{my_model.ligand_length} nm ligand length'] = [list(range(0, my_model.time + 1)), my_model.coverage] ###########
#         # data[f'{my_model.number_of_receptors} receptors'] = [list(range(0, my_model.time + 1)), my_model.coverage]
#         data[f'{my_model.receptor_length} nm receptor length'] = [list(range(0, my_model.time + 1)), my_model.coverage]
#         '''Plot log(Time) (Seconds) v surface coverage'''
#         # plt.subplot()
#         # plt.title(f'{my_model.number_of_nanoparticles} Nanoparticles, 10^{i} Ligands, {my_model.number_of_receptors} Receptors, Receptor length {my_model.receptor_length} nm')
#         # plt.xlabel('log(Time (seconds))')  # For log x-axis
#         # plt.ylabel('Surface Coverage')
#         # plt.plot([np.log10(i) for i in list(range(1, my_model.time+1))], my_model.coverage, label=f'Ligand length {my_model.ligand_length} nm')  # For log x-axi
#         '''Saving Data'''
#         # dictionary of lists
#         # dictionary = {'Time (Seconds)': list(range(1, my_model.time+1)), 'Surface Coverage': my_model.coverage}
#         # dictionary = {'log(Time) (Seconds)': [np.log10(i) for i in list(range(1, my_model.time+1)], 'Surface Coverage': b}  # For log x-axis
#         # df = pd.DataFrame(dictionary)
#         # saving the DataFrame
#         # df.to_csv('data12.CSV', index=False)
#         # my_model.nanoparticles_df.to_csv('data12_nanoparticles.CSV', index=False)
#         # my_model.receptors_df.to_csv('data12_receptors.CSV', index=False)
#     # plt.title(f'{int(my_model.dimension/1000)} μm\u00b3 system, {my_model.binding_energy} binding energy {my_model.number_of_nanoparticles} Nanoparticles,\n'
#     #           f'{my_model.nanoparticle_radius} nm Nanoparticle Radius, {my_model.number_of_ligands} Ligands, Ligand length {my_model.ligand_length} nm,\n'
#     #           f'{my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length')
#     plt.xlabel('Time (milliseconds)')
#     plt.ylabel('Surface Coverage')
#     for key, value in data.items():
#         plt.plot(value[0], value[1], label=key)
#     # leg = plt.legend(loc='best', ncol=2)
#     # leg.get_frame().set_alpha(0.5)
#     plt.legend()
#     plt.show()
#     second_variable_plot('Receptor length (nm)', 'Surface Coverage', d, final_surface_coverage)


# experiment_time()


# def experiment_variable():
#     coverage = [0]
#     for i in c:
#         number_of_seconds = 200  # i.e. 1 hour = 3600 seconds
#         my_model = MyModel(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=1000, receptor_length=100,
#                            number_of_nanoparticles=190, number_of_ligands=int(round(10 ** 2, 0)), nanoparticle_radius=50, ligand_length=7, binding_distance=i)
#         my_model.create_receptors()  # 100 nm for receptor
#         my_model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
#         print(f'{my_model.dimension} nm\u00b3 system, {my_model.binding_energy} binding energy\n'
#               f'{my_model.number_of_nanoparticles} Nanoparticles, {my_model.nanoparticle_radius} nm Nanoparticle Radius, {my_model.number_of_ligands} Ligands,\n'
#               f'Ligand length {my_model.ligand_length} nm, {my_model.number_of_receptors} Receptors, {my_model.receptor_length} nm Receptor length')
#         my_model.run(steps=number_of_seconds)  # 3600 for 1 hour
#         print(f'There were {my_model.count} reactions')
#         print(f'The surface coverage is {my_model.surface_coverage}')
#         coverage.append(my_model.surface_coverage)
#     '''Surface coverage v Variable value'''
#     plt.title(f'{my_model.dimension} nm\u00b3 system, {my_model.binding_energy} binding energy\n'
#               f'{my_model.number_of_nanoparticles} Nanoparticles, {my_model.number_of_ligands} Ligands, {my_model.ligand_length} nm Ligand length \n'  # {my_model.number_of_receptors} Receptors,
#               f'{my_model.receptor_length} nm Receptor length')
#     plt.xlabel('Number of Receptors')
#     plt.ylabel('End Surface Coverage')
#     c.insert(0, 0)
#     plt.plot(c, coverage)
#     plt.show()


# experiment_variable()

# attempts_for_mean = 1
# for i in a:
#     print(f'Number of Ligands = 10^{i}')
#     surface_coverage = 0
#     for j in range(attempts_for_mean):
#         my_model = MyModel(dimension=1000)
#         my_model.create_nanoparticles_and_ligands(number_of_nanoparticles=10, number_of_ligands=int(round(10 ** i, 0)), nanoparticle_radius=50, ligand_length=2)  # 1-2 nm for ligand
#         print('Created Nanoparticles')
#         my_model.create_receptors(number_of_receptors=1000, receptor_length=100)  # 100 nm for receptor
#         print('Created Receptors')
#         print('--Running model--')
#         my_model.run(steps=5)  # 3600 for 1 hour
#         print(f'There were {my_model.count} reactions')
#         # print(f'The surface coverage is {my_model.surface_coverage}')
#         surface_coverage += my_model.surface_coverage
#     mean_surface_coverage = surface_coverage/attempts_for_mean
#     print(f'The mean surface coverage is {mean_surface_coverage}')
#     b.append(mean_surface_coverage)  # Mean surface coverage

#
# # dictionary of lists
# dictionary = {'Log(Number of Ligands)': a, 'Surface Coverage': b}
# df = pd.DataFrame(dictionary)
# # saving the DataFrame
# df.to_csv('data6.CSV', index=False)
# print(df)
#
# # Plot Log(ligands) v surface coverage
# plt.xlabel('Log(Number of Ligands)')
# plt.ylabel('Surface Coverage')
# plt.plot(a, b)
# plt.show()
#
# print('Log Bases: ', a)
# print('Surface Coverages: ', b)
