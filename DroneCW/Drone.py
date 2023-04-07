import time
import random

class droneClient(object):
    """docstring for Drone."""

    def __init__(self, id):
        self.droneID = id
        self.home_position = None
        self.current_velocity = [0,0,0]
        self.current_position = [0,0,0]
        self.fly_new_position = [0,0,0]
        self.personal_best_position = [0,0,0]
        self.current_best = None
        self.personal_best = None
        self.err_best_i=-1          # best error individual
        self.err_i=-1

    def PDRONE(self):
        print(self.current_position)
        print(self.personal_best_position)
        print(self.current_best)
        print(self.personal_best)
        print(self.current_velocity)

    def evaluate(self, fitness_function):
        self.current_best = fitness_function(self.current_position)
        if self.current_best < self.personal_best:
            self.personal_best = self.current_best
#             self.personal_best_position = self.current_position

    def swarmUpdatePosVel(self, swarm_best):
        self.velocityUpdate(swarm_best)
        self.positionNewPosition()

    def positionNewPosition(self):
        for i in range(3):
            self.current_position[i] += self.current_velocity[i]

    def velocityUpdate(self, swarm_best, w_min=0.5, max=1.0, c=0.1):
        # Initialise new velocity array
        # Randomly generate r1, r2 and inertia weight from normal distribution
        r1 = random.uniform(0.01,1.0)
        r2 = random.uniform(0.01,1.0)
        w = random.uniform(w_min,max)
        c1 = c
        c2 = c
        new_velocity = [0.0]*3
        # Calculate new velocity

        for i in range(3):
            inn = w*self.current_velocity[i]
            a = c1*r1*(self.personal_best_position[i] - self.current_position[i])
            b = c2*r2*(swarm_best[i] - self.current_position[i])
            new_velocity[i] = inn + a + b

        if self.current_position[2] > -45:
            new_velocity[2] = -5.5
        self.current_velocity = new_velocity

class PSO(object):
    """docstring for PSO."""

    def __init__(self):
        super(PSO, self).__init__()
        self.drone_list = []
        self.gbest_value = None
        self.gbest_position = [0,0,0]
        self.av_bestfit = 0
        self.swarm_best = [None,None,None]
        self.start()
        self.run()
        self.psoUpdate()

    def fitness_function(self, dronePos):
        x1 = dronePos[0]
        x2 = dronePos[1]
        f1=x1+2*-x2+3
        f2=2*x1+x2-8
        z = f1**2+f2**2
        return z


    def start(self):
        #self.client.confirmConnection()
        id = "Drone"
        for i in range(6):
            id = "Drone" + str(i+1)
            print(id)
            newDrone = droneClient(id)
            self.drone_list.append(newDrone)

    def run(self):
        print('innit')
        for i in self.drone_list:
            x = random.uniform(-100, 100)
            y = random.uniform(-100, 100)
            i.current_position = [x,y,50]
            i.personal_best_position = i.current_position

        for i in self.drone_list:
            i.current_best = self.fitness_function(i.current_position)
            i.personal_best = i.current_best
            i.PDRONE()

        self.gbest_value = self.drone_list[0].personal_best
        self.gbest_position = self.drone_list[0].personal_best_position

        for i in self.drone_list:
            self.swarmEvaluate(i)

        self.avFitnessCalc()
        print(self.gbest_value)
        print(self.gbest_position)
        print(self.av_bestfit)



        print("done innit")
        #self.velocity_set()

        #innit done

    def velocity_set(self):
        print("innit velocity")
        for i in self.drone_list:
            r1 = random.uniform(-15, 15)
            r2 = random.uniform(-15, 15)
            i.current_velocity = [r1,r2,-2]


    def psoUpdate(self):
        for _ in range(0,400):
            if self.av_bestfit <= 10e-4:
                print("done")
                break
            else:
                for i in self.drone_list:
                    i.swarmUpdatePosVel(self.gbest_position)
                    # cycle through particles in swarm and evaluate fitness
                    i.evaluate(self.fitness_function)
                    # determine if the best in swarm


#                 self.gbest_value = self.drone_list[0].personal_best
#                 self.gbest_position = self.drone_list[0].personal_best_position

#                 for i in self.drone_list:
#                     self.swarmEvaluate(i)
                pbest_fitness = []
                for i in self.drone_list:
                    pbest_fitness.append(i.current_best)


                # Find the index of the best particle
                gbest_index = np.argmin(pbest_fitness)
                # Update the position of the best particle
                self.gbest_position = self.drone_list[gbest_index].personal_best_position
                self.gbest_value = pbest_fitness[gbest_index]

                self.avFitnessCalc()

                print(self.gbest_value,
                    self.gbest_position,
                    self.av_bestfit)

    def avFitnessCalc(self):
        av_best = 0
        for i in self.drone_list:
            av_best += i.personal_best
        self.av_bestfit = av_best/len(self.drone_list)

    def swarmEvaluate(self, drone):
        if drone.current_best < self.gbest_value:
            self.gbest_value = drone.personal_best
            self.gbest_position = drone.personal_best_position

if __name__ == '__main__':
    PSO()
