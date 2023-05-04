import airsim
import time
import random
import math
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image
from scipy.ndimage import gaussian_filter

class Client(object):
    client: airsim.MultirotorClient
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

    def enableApiControl(self, bool, id):
        self.client.enableApiControl(bool, id)

    def takeoffAsync(self, id):
        return self.client.takeoffAsync(vehicle_name = id)

    def simSetVehiclePose(self, pose, id):
        pose = airsim.Pose(airsim.Vector3r(pose[0], pose[1], pose[2]), airsim.to_quaternion(0, 0, 0))
        return self.client.simSetVehiclePose(pose, True, vehicle_name = id)

    def hoverAsync(self, id):
        return self.client.hoverAsync(vehicle_name = id)

    def getMultirotorState(self, id):
        return self.client.getMultirotorState(vehicle_name = id)

    def simSetTraceLine(self, id):
        self.client.simSetTraceLine([0.0, 1.0, 1.0, 1.0], thickness=10.0, vehicle_name=id)

    def moveToPositionAsync(self, position, velocity, camera_heading, id):
        return self.client.moveToPositionAsync(position[0], position[1], position[2], velocity, drivetrain=0, yaw_mode = airsim.YawMode(False, camera_heading), vehicle_name = id)

    def rotateByYawAsync(self, coordinates, id):
        vx, vy, vz = coordinates

        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        if speed > 0:
            yaw_rate = math.degrees(math.atan2(vy, vx))
        else:
            yaw_rate = 0

        # Turn the drone to face the direction of travel
        return self.client.rotateToYawAsync(yaw_rate, vehicle_name = id)

    def justturn(self, rot, id):
        return self.client.rotateToYawAsync(rot, vehicle_name = id)

    def getHomeGeoPoint(self, id):
        return self.client.getHomeGeoPoint(id)

    def goHomeAsync(self, id):
        return self.client.goHomeAsync(id)

    def moveByManualAsync(self, vx, vy, z_min, duration, id):
        return self.client.moveByManualAsync(vx, vy, z_min, duration, vehicle_name = id)

    def moveByVelocityAsync(self, vx, vy, vz, duration, id):
        # -vz = up +vz = down
        return self.client.moveByVelocityAsync(vx, vy, vz, duration, vehicle_name = id)

    def moveByVelocityZAsync(self, vx, vy, z, va, degree, settings, id):
        return self.client.moveByVelocityZAsync(vx, vy, z, va, degree, settings, vehicle_name = id)


    def takedroneimage(self, numId, droneID):
        responses = self.client.simGetImages([
        airsim.ImageRequest("f"+str(numId), airsim.ImageType.Scene)], droneID)
        #airsim.write_file(os.path.normpath('./images/fire_check_'+str(droneID)+'.png'), response.image_data_uint8)
        img = cv2.imdecode(np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img


    def takedroneimageOrbit(self, count):
        responses = self.client.simGetImages([
        airsim.ImageRequest("f1", airsim.ImageType.Scene)], "Drone1")

        for i, response in enumerate(responses):
                airsim.write_file(os.path.normpath('./images/drone1orbit' + str(count) +'.png'), response.image_data_uint8)#

        responses = self.client.simGetImages([
        airsim.ImageRequest("f2", airsim.ImageType.Scene)], "Drone2")

        for i, response in enumerate(responses):
                airsim.write_file(os.path.normpath('./images/drone2orbit' + str(count)+'.png'), response.image_data_uint8)

    def getDistanceSensorData(self, id):
        return self.client.getDistanceSensorData("Distance1", vehicle_name = id)

    def getLidarData(self, id, num):
        sensor = "l" + str(num)
        lidar_data = self.client.getLidarData(sensor, vehicle_name = id)
        if len(lidar_data.point_cloud) == 0:
            return False
        return True

    def dronePose(self, id):

        pose = self.client.simGetObjectPose(id)
        # extract the XYZ coordinates from the pose
        x = pose.position.x_val
        y = pose.position.y_val
        z = pose.position.z_val

        # print the XYZ coordinates
        print("Drone position: ({}, {}, {})".format(x, y, z))

class droneClient(object):
    """docstring for Drone."""

    def __init__(self, id, num):
        self.droneID = id
        self.droneNumID = num
        self.home_position = None
        self.position = np.array([0,0,0])
        self.fitness = 0
        self.best_fitness = None
        self.velocity = [0.0,0.0,0.0]
        self.pos_list = []
        self.search_position = self.position


    def PDRONE(self):
        #print(self.home_position)
        print(self.current_position)
        print(self.personal_best_position)
        print(self.current_best)
        print(self.personal_best)
        print(self.current_velocity)

    def local_search(self):
        self.best_position = self.position

    def fitness_calculation(self, target, radius):
        dist = np.linalg.norm(self.position - target) # Euclidean distance to target
        dist_from_edge = dist - radius # distance from edge of circle
        self.fitness = max(dist_from_edge, 0) # s
        if self.best_fitness != None:
            if self.fitness < self.best_fitness:
                self.best_position = self.position
                self.best_fitness = self.fitness
        else:
            self.best_fitness = self.fitness

    def velocity_update(self, drone_list, gbest_position, lidar_check, w_min=0.5, max=1.0, c=0.1):
        # Randomly generate r1, r2 and inertia weight from normal distribution
        collision_dist = 5
        r1 = random.uniform(0,max)
        r2 = random.uniform(0,max)
        w = random.uniform(w_min,max)
        c1 = c
        c2 = c

        avoidance_velocity = [0,0,0]
        for drone in drone_list:
            if drone.droneNumID != self.droneNumID:
                dist = np.linalg.norm(self.position - drone.position)
                if dist < collision_dist:
                    repulsive_force = (self.position - drone.position) / dist**2
                    avoidance_velocity += repulsive_force # add avoidance velocity to total

            new_velocity = []
            for i in range(len(self.velocity)):
                if i == 2:
                    new_velocity_i = w * self.velocity[i] + c1 * r1 * (self.best_position[i] - self.position[i]) \
                    + c2 * r2 * (gbest_position[i] - self.position[i]) + avoidance_velocity[i]
                else:
                    new_velocity_i = w * self.velocity[i] + c1 * r1 * (self.best_position[i] - self.position[i]) \
                    + c2 * r2 * (gbest_position[i] - self.position[i]) + avoidance_velocity[i] + random.uniform(-0.1, 0.1)

                new_velocity.append(new_velocity_i)

            estimate = self.position + new_velocity


            if estimate[2] > -55:
                new_velocity[2] -= random.uniform(0.5, 1)

            if lidar_check:
                new_velocity[2] -= 2
            else:
                if estimate[2] < -60:
                    new_velocity[2] += random.uniform(0.5, 1)

        self.velocity = new_velocity

    def velocity_update_explore(self, drone_list, unexplored_points, position_min, position_max, lidar_check, w_min=0.5, max=1.0):
        # Randomly generate r1, r2 and inertia weight from normal distribution
        collision_dist = 5
        r1 = random.uniform(0.5,max)
        r2 = random.uniform(0,max)
        w = random.uniform(w_min,max-0.2)
        c1 = 0.8
        c2 = 0.15

        avoidance_velocity = [0,0,0]
        exploration_velocity = [0,0,0]

        # Calculate the avoidance velocity based on nearby drones ########################### REDO #####################################
        for drone in drone_list:
            if drone.droneNumID != self.droneNumID:
                dist = np.linalg.norm(self.position - drone.position)
                if dist < collision_dist:
                    repulsive_force = (self.position - drone.position) / dist**2
                    avoidance_velocity += repulsive_force # add avoidance velocity to total

        # Calculate the exploration velocity based on the distance to the nearest unexplored point
        exploration_velocity = np.array((self.search_position[:2] - self.position[:2]) / np.linalg.norm(self.search_position[:2] - self.position[:2]))
        exploration_velocity = np.append(exploration_velocity, 0)
        new_velocity = []
        for i in range(len(self.velocity)):
            if i == 2:
                new_velocity_i = w * self.velocity[i] + c1 * r1 * exploration_velocity[i] + c2 * r2 * avoidance_velocity[i]
            else:
                new_velocity_i = w * self.velocity[i] + c1 * r1 * exploration_velocity[i] + c2 * r2 * avoidance_velocity[i] + random.uniform(-1, 1)

            new_velocity.append(new_velocity_i)

        # Check that the new position is within the boundaries of the search area
        estimate = self.position + new_velocity

        for i in range(len(estimate)-1):
            if estimate[i] < position_min[i]:
                new_velocity[i] += random.uniform(0.5, 1)
            elif estimate[i] > position_max[i]:
                new_velocity[i] -= random.uniform(0.5, 1)

        if estimate[2] > -55:
                new_velocity[2] -= random.uniform(0.5, 1)

        if lidar_check:
            new_velocity[2] -= 2
        else:
            if estimate[2] < -255: #height boundary for simulation
                new_velocity[2] += random.uniform(1, 1.5)

        self.velocity = new_velocity

    def position_update(self):
        self.position += self.velocity
        self.pos_list.append(np.array(self.position))


class PSO(object):
    """docstring for PSO."""

    def __init__(self):
        super(PSO, self).__init__()
        self.position_min = []
        self.position_max = []
        self.drone_list = []
        self.PSOclient = Client()
        self.target = None
        self.fitness_criterion = 10e-4
        self.gbest_fitness = 0
        self.gbest_pos = []
        self.search_radius = 25
        self.heightmap_generate()
        self.start()
        self.logic_control()




    def angle_calc(self, coordinates):
        vx, vy, vz = coordinates
        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        if speed > 0:
            return math.degrees(math.atan2(vy, vx))
        return 0

    def heightmap_generate(self):
        # Load the heightmap from the PNG file
        img = Image.open('map_large.png').convert('L')
        heightmap = np.array(img)

        # Smooth the heightmap using a Gaussian filter
        self.heightmap = gaussian_filter(heightmap, sigma=3)
        print(self.heightmap.shape)
        self.position_min = [0,0]
        self.position_max = [self.heightmap.shape[1],self.heightmap.shape[0]]
        #Create a 3D plot
        # fig = plt.figure(figsize=(10,10))
        # ax = fig.add_subplot(projection='3d')
        #
        # # Set the x, y, and z limits
        # xlen = self.heightmap.shape[1]
        # ylen = self.heightmap.shape[0]
        # x = np.linspace(0, xlen - 1, xlen)
        # y = np.linspace(0, ylen - 1, ylen)
        # X, Y = np.meshgrid(x, y)
        # Z = self.heightmap #/ 255.0 # normalize the heights to [0,1]
        #
        # ax.set_xlim(0, xlen)
        # ax.set_ylim(ylen, 0)
        # ax.set_zlim(0, np.max(heightmap))
        #
        # # Create a surface plot
        # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
        #                        linewidth=0, antialiased=False, zorder=10,alpha=0.4)
        #
        # # Add a color bar
        # #fig.colorbar(surf, shrink=0.5, aspect=5)
        #
        # # Set the scaling of the axes
        # ax.set_box_aspect([1., 1, 0.2])
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Height')
        #
        # return ax

    def detect_fire(self, image):
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Define the lower and upper bounds for the "fire" color
        lower_red = np.array([0, 70, 50])
        upper_red = np.array([10, 255, 255])
        # Threshold the image to get the "fire" mask
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        # Define the lower and upper bounds for the "fire" color (again)
        lower_red = np.array([170, 70, 50])
        upper_red = np.array([180, 255, 255])
        # Threshold the image to get the "fire" mask (again)
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        # Combine the two masks
        mask = cv2.bitwise_or(mask1, mask2)
        # Apply a median blur to reduce noise
        mask = cv2.medianBlur(mask, 5)
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Calculate the total area of the fire-like regions
        if(len(contours) == 0):
            return 0
        else:
            areas = [cv2.contourArea(contour) for contour in contours]
            total_area = np.sum(areas)
            largest_contour = max(contours, key=cv2.contourArea)
            _, _, w, h = cv2.boundingRect(largest_contour)
            score = total_area / (w * h)
            return score

    def show_fire(self, image):
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Define the lower and upper bounds for the "fire" color
        lower_red = np.array([0, 70, 50])
        upper_red = np.array([10, 255, 255])
        # Threshold the image to get the "fire" mask
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        # Define the lower and upper bounds for the "fire" color (again)
        lower_red = np.array([170, 70, 50])
        upper_red = np.array([180, 255, 255])
        # Threshold the image to get the "fire" mask (again)
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        # Combine the two masks
        mask = cv2.bitwise_or(mask1, mask2)
        # Apply a median blur to reduce noise
        mask = cv2.medianBlur(mask, 5)
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Calculate the total area of the fire-like regions

        total_area = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            area = cv2.contourArea(contour)
            total_area += area

        largest_contour = max(contours, key=cv2.contourArea)
        _,_,w,h = cv2.boundingRect(largest_contour)
        score = total_area / (w*h)


        cv2.drawContours(image, [largest_contour], 0, (0, 255, 0), 2)

        # Show the image with the boxes and the prediction score
        cv2.putText(image, "Fire score: {:.2f}".format(score), (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("output", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    def start(self):
        #self.client.confirmConnection()
        id = "Drone"
        for i in range(20):
            id = "Drone" + str(i+1)
            print(id)
            self.PSOclient.enableApiControl(True, id)
            newDrone = droneClient(id, i+1)
            self.drone_list.append(newDrone)


        drone_pose = [0,-10,-2]
        for batch in range(2):
            start = batch * 10
            end = (batch + 1) * 10
            for i, drone in enumerate(self.drone_list[start:end]):
                drone_pose[1] += 2
                self.PSOclient.simSetVehiclePose(drone_pose, drone.droneID)
                time.sleep(0.21)
                if drone.droneNumID == end:
                    time.sleep(2)

        print("pose done")
        for batch in range(2):
            start = batch * 10
            end = (batch + 1) * 10
            for i, drone in enumerate(self.drone_list[start:end]):
                self.PSOclient.takeoffAsync(drone.droneID)
                time.sleep(0.1)
                if drone.droneNumID == end:
                    time.sleep(2)

        time.sleep(1)
        # print(self.PSOclient.getLidarData(self.drone_list[0].droneID, self.drone_list[0].droneNumID))
        # time.sleep(1)

        for i, drone in enumerate(self.drone_list):
            drone.home_position = self.PSOclient.getMultirotorState(drone.droneID).kinematics_estimated.position
            print(drone.home_position)
            time.sleep(0.1)
        #=

        time.sleep(2)

        target_innitHeight = -55
        init_height = [0, -10, -55]

        for batch in range(2):
            start = batch * 10
            end = (batch + 1) * 10
            for i, drone in enumerate(self.drone_list[start:end]):
                init_height[1] += 2
                init_height[2] += 2
                thread = self.PSOclient.moveToPositionAsync(init_height, 25, self.angle_calc(drone.position), drone.droneID)
                if drone.droneNumID == end:
                    thread.join()
                    #time.sleep(4)

        time.sleep(1)

        for i in self.drone_list:
            position = self.PSOclient.getMultirotorState(i.droneID).kinematics_estimated.position
            x = position.x_val
            y = position.y_val
            z = position.z_val
            i.position =  np.array([x, y, z])
            self.PSOclient.simSetTraceLine(i.droneID)
            time.sleep(0.1)

        time.sleep(2.5)


    def get_square_centers(self, min_coord, max_coord, divisions):
        # Calculate the width and height of each square
        width = max_coord[0] - min_coord[0]
        height = max_coord[1] - min_coord[1]

        rect_width = width // divisions
        rect_height = height // divisions

        # Calculate the coordinates of the top left corner of the first rectangle
        x_offset = rect_width // 2 + min_coord[0]
        y_offset = rect_height // 2 + min_coord[1]

        # Create a list to store the rectangle center coordinates
        rect_centers = []

        # Iterate over each row and column to calculate the center of each rectangle
        for row in range(divisions):
            for col in range(divisions):
                x = (row * rect_width + x_offset)
                y = (col * rect_height + y_offset)
                z = -55
                new_point = [float(x),float(y), float(z)]
                rect_centers.append(new_point)

        return rect_centers

    def explored_check(self):
        explored_points = set()
        unexplored_points = []
        for drone in self.drone_list:
            for point in self.centres:
                if np.linalg.norm(drone.position[:2] - point[:2]) < 15: # if within 15m of the waypoint, assume this location has been checked.
                    explored_points.add(tuple(point))

        for p in self.centres:
            if tuple(p) not in explored_points:
                unexplored_points.append(p)
        self.centres = unexplored_points


    def drone_explore_position(self):
        currently_exploring = set()
        minimum_dists = []
        for drone in self.drone_list:
            candidates = []
            for point in self.centres:
                if tuple(point) not in currently_exploring:
                    candidates.append(point)

            if not candidates: #if no available candidates then select closest even if already pursued by other drone
                min_dists = np.array([np.linalg.norm(drone.position[:2] - point[:2]) for point in self.centres])
                min_index = np.argmin(min_dists)
                drone.search_position = self.centres[min_index]
            else:
                min_dists = np.array([np.linalg.norm(drone.position[:2] - point[:2]) for point in candidates])
                min_index = np.argmin(min_dists)
                drone.search_position = candidates[min_index]
                currently_exploring.add(tuple(candidates[min_index]))


    def explore(self):
        print("explore logic")
        flag = False
        for t in range(250):
            print(t)
            if flag:
                print(t)
                break
            self.drone_explore_position()
            for drone in self.drone_list:
                drone.velocity_update_explore(self.drone_list, self.centres, self.position_min, self.position_max, self.PSOclient.getLidarData(drone.droneID, drone.droneNumID))
                drone.position_update()
                thread = self.PSOclient.moveToPositionAsync(drone.position, 5, self.angle_calc(drone.velocity), drone.droneID)
                #take image
                if drone.droneNumID == len(self.drone_list):
                    thread.join()

            for drone in self.drone_list:
                image = self.PSOclient.takedroneimage(drone.droneNumID, drone.droneID)
                score = self.detect_fire(image)
                if score > 0:
                    print('fire_detected')
                    self.show_fire(image)
                    self.target = drone.position
                    print(self.target)
                    flag = True
                    break

            print("\r" + str(len(self.centres)), end = " ")

            self.explored_check()
            if len(self.centres) == 0:
                print("out of search pos")
                flag = True
                break

    def single_step(self):
        for i in self.centres:
            thread = self.PSOclient.moveToPositionAsync(i, 5, self.angle_calc([0,0,0]), "Drone1")
            thread.join()



    def logic_control(self):

        print("start logic")
        print(self.position_min, self.position_max)
        self.centres = self.get_square_centers(self.position_min, self.position_max, 10)
        #self.explore()
        # print("end explore")
        # if self.target != None:
        #     self.run()


    def run(self):
        print('running')
        self.generation = 100

        for drone in self.drone_list:
            drone.local_search()

        for drone in self.drone_list:
            drone.fitness_calculation(self.target, self.search_radius)

        fitness = [drone.fitness for drone in self.drone_list]
        best_index = np.argmin(fitness)
        self.gbest_fitness = fitness[best_index]
        self.gbest_pos = self.drone_list[best_index].position
        print(self.gbest_fitness, self.gbest_pos)

        for t in range(self.generation):
            print(t)
            if np.average([drone.fitness for drone in self.drone_list]) <= self.fitness_criterion:
                print("fit ", t)
                break
            else:
                for drone in self.drone_list:
                    drone.fitness_calculation(self.target, self.search_radius)
                    drone.velocity_update(self.drone_list, self.gbest_pos, self.PSOclient.getLidarData(drone.droneID, drone.droneNumID))
                    drone.position_update()
                    thread = self.PSOclient.moveToPositionAsync(drone.position, 5, self.angle_calc(drone.velocity), drone.droneID)
                    time.sleep(0.1)
                    #take image
                    if drone.droneNumID == len(self.drone_list):
                        thread.join()


                fitness = [drone.fitness for drone in self.drone_list]
                best_index = np.argmin(fitness)
                self.gbest_fitness = fitness[best_index]
                self.gbest_pos = self.drone_list[best_index].position


    def track_orbits(self, angle):
        diff = abs(angle - self.start_angle)
        if self.start_flag == None:
            if (diff < 1):
                self.start_flag = True
            else:
                print("not innit")
            return False
        else:
            if round(diff) >= 358:
                return True
            else:
                if round(diff) > self.prev_angle:
                    self.PSOclient.takedroneimageOrbit(self.prev_angle)
                    self.prev_angle += 90
                    print("1/4")

                return False

    async def orbit(self): # make async check this works
        print("ramping up to speed...")
        count = 0

        radius = 5
        # ramp up time
        ramptime = radius / 10
        start_time = time.time()

        center = [1,0]
        cx = float(center[0])
        cy = float(center[1])
        length = math.sqrt((cx*cx) + (cy*cy))
        cx /= length
        cy /= length
        cx *= radius
        cy *= radius

        pos_start = self.getCurrentPosition("Drone2")
        center = pos_start
        center.x_val += cx
        center.y_val += cy

        self.start_angle = -180
        self.start_flag = None
        self.prev_angle = 87
        while count < 1:
            # ramp up to full speed in smooth increments so we don't start too aggressively.
            now = time.time()
            speed = 2
            speed_const = 2
            diff = now - start_time
            if diff < ramptime:
                speed = speed_const * diff / ramptime
            elif ramptime > 0:
                print("reached full speed...")
                ramptime = 0

            lookahead_angle = speed / radius

            # compute current angle
            pos = self.getCurrentPosition("Drone2")
            dx = pos.x_val - center.x_val
            dy = pos.y_val - center.y_val
            actual_radius = math.sqrt((dx*dx) + (dy*dy))
            angle_to_center = math.atan2(dy, dx)

            camera_heading = (angle_to_center - math.pi) * 180 / math.pi

            # compute lookahead
            lookahead_x = center.x_val + radius * math.cos(angle_to_center + lookahead_angle)
            lookahead_y = center.y_val + radius * math.sin(angle_to_center + lookahead_angle)

            vx = lookahead_x - pos.x_val
            vy = lookahead_y - pos.y_val

            if self.track_orbits(angle_to_center * 180 / math.pi):
                count += 1
                print("completed {} orbits".format(count))

            self.PSOclient.moveByVelocityZAsync(vx, vy, -25, 2, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, camera_heading), "Drone2")

    def psoUpdate(self):
        for _ in range(0,50):
            if self.av_bestfit <= 10e-4:
                print("done")
                break
            else:
                for i in self.drone_list:
                    i.swarmUpdatePosVel(self.gbest_position)
                    # cycle through particles in swarm and evaluate fitness
                    i.evaluate(self.fitness_function)
                    # determine if the best in swarm
                    self.droneRotateCallAsunc(i.current_velocity, i.droneID)
                    print("r")
                    time.sleep(1)
                    self.droneVelocityCallAsync(i.current_position, i.droneID)


                time.sleep(5)

                itrbest_fitness = self.drone_list[0].current_best
                index = 0
                for i in self.drone_list:
                    if i.current_best < itrbest_fitness:
                        itrbest_fitness = i.current_best
                        index = self.drone_list.index(i)

                self.gbest_position = self.drone_list[index].personal_best_position
                self.gbest_value = self.drone_list[index].current_best

                self.avFitnessCalc()

                print(self.gbest_value,
                    self.gbest_position,
                    self.av_bestfit)

    def avFitnessCalc(self):
        av_best = 0
        for i in self.drone_list:
            av_best += i.personal_best
        self.av_bestfit = av_best/len(self.drone_list)

    def droneVelocityCallAsync(self, fly_new_position, ID):

        self.PSOclient.moveToPositionAsync(
            fly_new_position[0],
            fly_new_position[1],
            fly_new_position[2], 5, ID)

    def droneRotateCallAsunc(self, velocity, ID):
        self.PSOclient.rotateByYawRateAsync(velocity, ID)

    def droneVelocityCall(self, fly_new_position, ID):

        thread = self.PSOclient.moveToPositionAsync(
            fly_new_position[0],
            fly_new_position[1],
            fly_new_position[2], 5, ID)

        thread.join()


    def swarmEvaluate(self, drone):
        if drone.current_best < self.gbest_value:
            self.gbest_value = drone.personal_best
            self.gbest_position = drone.personal_best_position


    def getCurrentPosition(self, drone):
        current_position = self.PSOclient.getMultirotorState(drone)
        # print(current_position.gps_location)
        # print(current_position.kinematics_estimated.position) #local to the spaw area
        # print('velocity', current_position.kinematics_estimated.angular_velocity)
        # print('acceleration', current_position.kinematics_estimated.angular_velocity)
        return current_position.kinematics_estimated.position

if __name__ == '__main__':
    PSO()
