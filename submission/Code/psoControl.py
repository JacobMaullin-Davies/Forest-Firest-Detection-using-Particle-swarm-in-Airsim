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
from scipy.interpolate import interp2d
import multiprocessing as mp
import threading

class Client(object):
    """ API client """
    client: airsim.MultirotorClient
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

    def enableApiControl(self, bool, id):
        """ Api control """
        self.client.enableApiControl(bool, id)

    def takeoffAsync(self, id):
        """ Take off """
        return self.client.takeoffAsync(vehicle_name = id)

    def simSetVehiclePose(self, pose, id):
        """ Set pose of drone """
        pose = airsim.Pose(airsim.Vector3r(pose[0], pose[1], pose[2]), airsim.to_quaternion(0, 0, 0))
        return self.client.simSetVehiclePose(pose, True, vehicle_name = id)

    def hoverAsync(self, id):
        """ Set hover """
        return self.client.hoverAsync(vehicle_name = id)

    def getMultirotorState(self, id):
        """ Get drone state """
        return self.client.getMultirotorState(vehicle_name = id)

    def simSetTraceLine(self, id):
        """ Set trace line """
        self.client.simSetTraceLine([0.0, 1.0, 1.0, 1.0], thickness=20.0, vehicle_name=id)

    def moveToPositionAsync(self, position, velocity, camera_heading, id):
        """ Position update  """
        return self.client.moveToPositionAsync(position[0], position[1], position[2], velocity, drivetrain=0, yaw_mode = airsim.YawMode(False, camera_heading), vehicle_name = id)

    def rotateByYawAsync(self, coordinates, id):
        """ Rotate drone """
        vx, vy, vz = coordinates

        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        if speed > 0:
            yaw_rate = math.degrees(math.atan2(vy, vx))
        else:
            yaw_rate = 0

        # Turn the drone to face the direction of travel
        return self.client.rotateToYawAsync(yaw_rate, vehicle_name = id)

    def justturn(self, rot, id):
        """ Only turn drone """
        return self.client.rotateToYawAsync(rot, vehicle_name = id)

    def getHomeGeoPoint(self, id):
        """ Home point """
        return self.client.getHomeGeoPoint(id)

    def goHomeAsync(self, id):
        """ Return home  """
        return self.client.goHomeAsync(id)

    def moveByManualAsync(self, vx, vy, z_min, duration, id):
        """  Manual movement """
        return self.client.moveByManualAsync(vx, vy, z_min, duration, vehicle_name = id)

    def moveByVelocityAsync(self, vx, vy, vz, duration, id):
        """ Velocity update move """
        # -vz = up +vz = down
        return self.client.moveByVelocityAsync(vx, vy, vz, duration, vehicle_name = id)

    def moveByVelocityZAsync(self, vx, vy, z, va, degree, settings, id):
        """ Z only movement """
        return self.client.moveByVelocityZAsync(vx, vy, z, va, degree, settings, vehicle_name = id)


    def takedroneimage(self, numId, droneID):
        """ Drone image caputure  """
        responses = self.client.simGetImages([
        airsim.ImageRequest("f"+str(numId), airsim.ImageType.Scene)], droneID)
        img = cv2.imdecode(np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img


    def takedroneimageOrbit(self, count):
        """ Orbit test """
        responses = self.client.simGetImages([
        airsim.ImageRequest("f1", airsim.ImageType.Scene)], "Drone1")

        for i, response in enumerate(responses):
                airsim.write_file(os.path.normpath('./images/drone1orbit' + str(count) +'.png'), response.image_data_uint8)#

        responses = self.client.simGetImages([
        airsim.ImageRequest("f2", airsim.ImageType.Scene)], "Drone2")

        for i, response in enumerate(responses):
                airsim.write_file(os.path.normpath('./images/drone2orbit' + str(count)+'.png'), response.image_data_uint8)

    def getDistanceSensorData(self, id):
        """ Front distance sensor """
        return self.client.getDistanceSensorData("Distance1", vehicle_name = id)

    def getLidarData(self, id, num):
        """ Lidar data """
        sensor = "l" + str(num)
        lidar_data = self.client.getLidarData(sensor, vehicle_name = id)
        if len(lidar_data.point_cloud) == 0:
            return False
        return True

    def dronePose(self, id):
        """ Set drone pose """

        pose = self.client.simGetObjectPose(id)
        # extract the XYZ coordinates from the pose
        x = pose.position.x_val
        y = pose.position.y_val
        z = pose.position.z_val

        # print the XYZ coordinates
        print("Drone position: ({}, {}, {})".format(x, y, z))

    def simPlotPoints(self, points):
        """ Show points """
        position = [airsim.Vector3r(points[0], points[1], points[2])]
        self.client.simPlotPoints(position, color_rgba=[1.0, 0.0, 0.0, 1.0], size=10.0, duration=15.0, is_persistent=False)


class droneClient(object):
    """ Drone client obejct"""

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
        self.search_waypoints = []
        self.fire_angle = []
        self.fire_position = []
        self.best_score_v = 0
        self.best_im_v = None


    def PDRONE(self):
        #print(self.home_position)
        print(self.current_position)
        print(self.personal_best_position)
        print(self.current_best)
        print(self.personal_best)
        print(self.current_velocity)

    def local_search(self):
        """ Set best positon """
        self.best_position = self.position

    def fitness_calculation(self, target, radius):
        """ Fitness calcualtion """
        dist = np.linalg.norm(self.position[:2] - target[:2]) # Euclidean distance to target
        dist_from_edge = dist - radius # distance from edge of circle
        self.fitness = max(dist_from_edge, 0) # fitness
        if self.best_fitness != None:
            if self.fitness < self.best_fitness:
                self.best_position = self.position
                self.best_fitness = self.fitness
        else:
            self.best_fitness = self.fitness

    def velocity_update(self, drone_list, gbest_position, lidar_check, w_min=0.5, max=1.0, c=0.1):
        """ Drone velocity update """
        # Randomly generate r1, r2 and inertia weight from normal distribution
        collision_dist = 10
        r1 = random.uniform(0.1,max)
        r2 = random.uniform(0.1,max)
        w = random.uniform(w_min,max)
        c1 = 0.75
        c2 = 0.35

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
                + c2 * r2 * (gbest_position[i] - self.position[i]) + 0.5 * avoidance_velocity[i]
            else:
                new_velocity_i = w * self.velocity[i] + c1 * r1 * (self.best_position[i] - self.position[i]) \
                + c2 * r2 * (gbest_position[i] - self.position[i]) + 0.5 * avoidance_velocity[i] + random.uniform(-0.5, 0.5)

            new_velocity.append(new_velocity_i)

        estimate = self.position + new_velocity

        for drone in drone_list:
            if drone.droneNumID != self.droneNumID:
                dist = np.linalg.norm(estimate - drone.position)
                if dist < collision_dist:
                    repulsive_force = (estimate - drone.position) / dist**2
                    avoidance_velocity += repulsive_force # add avoidance velocity to total

        new_velocity += 0.4 * np.array(avoidance_velocity)


        if estimate[2] > -45:
            new_velocity[2] -= random.uniform(0.5, 1)

        if lidar_check:
            new_velocity[2] -= 2
        else:
            if estimate[2] < -80:
                new_velocity[2] += random.uniform(0.5, 1)

        self.velocity = new_velocity

    def velocity_update_explore(self, drone_list, unexplored_points, position_min, position_max, lidar_check, w_min=0.5, max=1.0):
        """ Exploration update. exp_w designates how much to explore towards waypoint target"""
        # Randomly generate r1, r2 and inertia weight from normal distribution
        collision_dist = 10
        r1 = random.uniform(0.5,max+1)
        r2 = random.uniform(0,max)
        w = random.uniform(w_min,max-0.2)
        c1 = 0.75
        c2 = 0.35

        avoidance_velocity = [0,0,0]

        # Calculate the avoidance velocity based on nearby drones
        for drone in drone_list:
            if drone.droneNumID != self.droneNumID:
                dist = np.linalg.norm(self.position - drone.position)
                if dist < collision_dist:
                    repulsive_force = (self.position - drone.position) / dist**2
                    avoidance_velocity += repulsive_force # add avoidance velocity to total

        # Calculate the exploration velocity based on the distance to the nearest unexplored point
        exploration_velocity = np.array((self.search_position - self.position) / np.linalg.norm(self.search_position - self.position)) * 2.5
        new_velocity = []
        for i in range(len(self.velocity)):
            if i == 2:
                new_velocity_i = w * self.velocity[i] + c1 * r1 * exploration_velocity[i] + c2 * r2 * avoidance_velocity[i]
            else:
                new_velocity_i = w * self.velocity[i] + c1 * r1 * exploration_velocity[i] + c2 * r2 * avoidance_velocity[i] + random.uniform(0.1, 0.5)

            new_velocity.append(new_velocity_i)


        estimate = self.position + new_velocity

        for drone in drone_list:
            if drone.droneNumID != self.droneNumID:
                dist = np.linalg.norm(estimate - drone.position)
                if dist < collision_dist:
                    repulsive_force = (estimate - drone.position) / dist**2
                    avoidance_velocity += repulsive_force # add avoidance velocity to total

        new_velocity += 0.2 * np.array(avoidance_velocity)
        # Check that the new position is within the boundaries of the search area

        for i in range(len(estimate)-1):
            if estimate[i] < position_min[i]:
                new_velocity[i] += random.uniform(0.5, 1)
            elif estimate[i] > position_max[i]:
                new_velocity[i] -= random.uniform(0.5, 1)

        if estimate[2] > -45:
                new_velocity[2] -= random.uniform(0.5, 1)

        if lidar_check:
            new_velocity[2] -= 2
        else:
            if estimate[2] < -180: #height boundary for simulation
                new_velocity[2] += random.uniform(1, 1.5)

        self.velocity = new_velocity

    def position_update(self):
        """ Position update """
        self.position += self.velocity
        self.pos_list.append(np.array(self.position))

    def position_update_moveW(self, position):
        self.position = np.array(position)


class PSO(object):
    """PSO operation object"""

    def __init__(self):
        super(PSO, self).__init__()
        self.position_min = []
        self.position_max = []
        self.drone_list = []
        self.PSOclient = Client()
        self.target = None
        self.score = 0.0
        self.fitness_criterion = 10e-4
        self.gbest_fitness = 0
        self.gbest_pos = []
        self.search_radius = 40
        self.filename = "Fire_location.txt"
        self.clear_file()
        self.fig3d = self.heightmap_generate()
        self.display_centres()
        self.start()
        self.logic_control()

    def display_centres(self):
        """ Show waypoints on map """
        #assuming that heightmap_generate() has been run already
        self.centres = self.get_square_centers(self.position_min, self.position_max, 20)
        for i in self.centres:
            self.PSOclient.simPlotPoints(i)


    def clear_file(self):

        with open(self.filename, 'w') as file:
            file.truncate(0)
        print(f"The contents of '{self.filename}' have been cleared.")

        with open("Fire_l_records.txt", 'w') as file:
            file.truncate(0)

    def read_fire_file(self):

        with open(self.filename, 'r') as file:
            first_line = file.readline().strip()

        if first_line == '':
            return [False, [], 0.0]

        else:
            array_str, score_str = first_line.split(' | ')
            array = np.array(list(map(float, array_str.split())))
            score = float(score_str)

        return [True, array, score]

    def read_all_fire_l(self):

        with open("Fire_l_records.txt", 'r') as file:
            lines = file.readlines()

        arrays = []
        for line in lines:
            line = line.strip()
            array_str = line[1:-1]
            array = np.array(list(map(float, array_str.split())))
            arrays.append(array)
        return np.stack(arrays)


    def angle_calc(self, coordinates):
        """ direction to move calculate """
        vx, vy, vz = coordinates
        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        if speed > 0:
            return math.degrees(math.atan2(vy, vx))
        return 0

    def heightmap_generate(self):
        """ heightmap generate """
        # Load the heightmap from the PNG file
        img = Image.open('small_hmp.png').convert('L')

        heightmap = np.array(img)

        # Smooth the heightmap using a Gaussian filter
        self.heightmap = gaussian_filter(heightmap, sigma=3)
        print(self.heightmap.shape)
        self.position_min = [0,0]
        self.position_max = [self.heightmap.shape[1],self.heightmap.shape[0]]

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')

        # Set the x, y, and z limits
        xlen = self.heightmap.shape[1]
        ylen = self.heightmap.shape[0]
        x = np.linspace(0, xlen - 1, xlen)
        y = np.linspace(0, ylen - 1, ylen)
        X, Y = np.meshgrid(x, y)
        Z = self.heightmap #/ 255.0 # normalize the heights to [0,1]

        ax.set_xlim(0, xlen)
        ax.set_ylim(ylen, 0)
        ax.set_zlim(0, np.max(heightmap))

        # Create a surface plot
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False, zorder=10,alpha=0.4)

        # Add a color bar
        #fig.colorbar(surf, shrink=0.5, aspect=5)

        # Set the scaling of the axes
        ax.set_box_aspect([1., 1, 0.2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Height')
        ax.set_xlim(self.position_max[0], self.position_min[0])
        ax.set_ylim(self.position_min[1], self.position_max[1])

        #plt.show()

        return ax

    def detect_fire(self, image):
        """ Check fire in image"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Define the lower and upper bounds for the "fire" color
        color_ranges = [
            #((5, 50, 50), (20, 255, 255)),
            ((0, 160, 135), (18, 255, 255)),
            #((25, 150, 20), (30, 255, 255))
            ]

        #((0, 100, 20), (13, 255, 255)),
        #
        mask_arry = []

        for lower_bound, upper_bound in color_ranges:
            # Create a mask for the current color range
            mask_arry.append(cv2.inRange(hsv, np.array(lower_bound), np.array(upper_bound)))


        combined_mask = np.zeros_like(mask_arry[0], dtype=np.uint8)


        # Iterate over the mask arrays and combine them using bitwise OR
        for mask in mask_arry:
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        combined_mask = combined_mask.astype(np.uint8)
        combined_mask = cv2.medianBlur(combined_mask, 5)

        result = cv2.bitwise_and(image, image, mask=combined_mask)

        mask1 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        mask = cv2.medianBlur(mask1, 5)
        # plt.imshow(mask)
        # plt.show()

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if(len(contours) == 0):
            return 0
        else:
            areas = [cv2.contourArea(contour) for contour in contours]
            total_area = np.sum(areas)
            largest_contour = max(contours, key=cv2.contourArea)
            _, _, w, h = cv2.boundingRect(largest_contour)
            score = total_area / (w * h)
            # self.show_fire(image)
            return score

    def show_fire(self, image):
        """ Show fire in image """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Define the lower and upper bounds for the "fire" color
        color_ranges = [
            #((5, 50, 50), (20, 255, 255)),
            ((0, 160, 135), (18, 255, 255)),
            #((25, 150, 20), (30, 255, 255))
            ]
        #
        mask_arry = []

        for lower_bound, upper_bound in color_ranges:
            # Create a mask for the current color range
            mask_arry.append(cv2.inRange(hsv, np.array(lower_bound), np.array(upper_bound)))


        combined_mask = np.zeros_like(mask_arry[0], dtype=np.uint8)

        # Iterate over the mask arrays and combine them using bitwise OR
        for mask in mask_arry:
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        combined_mask = combined_mask.astype(np.uint8)
        combined_mask = cv2.medianBlur(combined_mask, 5)

        result = cv2.bitwise_and(image, image, mask=combined_mask)

        mask1 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        mask = cv2.medianBlur(mask1, 5)
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
        # Calculate the total area of the fire-like regions
            total_area = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                area = cv2.contourArea(contour)
                total_area += area
        #
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
        """ Start operation """
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
                thread = self.PSOclient.moveToPositionAsync(init_height, 25, self.angle_calc(drone.position), drone.droneID)
                time.sleep(0.15)
                if drone.droneNumID == end:
                    thread.join()
                    #time.sleep(4)
        print("innit height done")
        time.sleep(1)

        for drone in self.drone_list:
            position = self.PSOclient.getMultirotorState(drone.droneID).kinematics_estimated.position
            x = position.x_val
            y = position.y_val
            z = position.z_val
            drone.position = np.array([x, y, z])
            drone.pos_list.append(np.array(drone.position))
            self.PSOclient.simSetTraceLine(drone.droneID)
            time.sleep(0.1)

        time.sleep(2.5)



    def get_square_centers(self, min_coord, max_coord, divisions):
        """ Generate waypoints """
        # Calculate the width and height of each square

        # max_coord = [2000,1400]


        width = max_coord[0] - min_coord[0]
        height = max_coord[1] - min_coord[1]

        print(width, height)

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
                z =  40 +    (-1 * self.heightmap[y,x])
                new_point = [float(x),float(y), float(z)]
                rect_centers.append(new_point)

        return rect_centers

    def explored_check(self):
        """ Check waypoint distance """
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
        """ explore set distance """
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
        """ Explore logic """
        print("explore logic")
        flag = False
        for t in range(4000):
            if flag:
                print(t)
                break
            result_check = self.read_fire_file()
            if result_check[0]:
                print("fire", result_check[1])
                self.target = result_check[1]
                self.score  = result_check[2]
                flag = True
                break

            self.drone_explore_position()
            for drone in self.drone_list:
                drone.velocity_update_explore(self.drone_list, self.centres, self.position_min, self.position_max, self.PSOclient.getLidarData(drone.droneID, drone.droneNumID))
                drone.position_update()
                thread = self.PSOclient.moveToPositionAsync(drone.position, 4.5, self.angle_calc(drone.velocity), drone.droneID)
                time.sleep(0.1)
                if drone.droneNumID == len(self.drone_list):
                    thread.join()

            print("\r" + "iteration: "+ str(t) + " search areas remaining: " + str(len(self.centres)), end = " ")

            self.explored_check()
            if len(self.centres) == 0:
                print("out of search pos")
                flag = True
                break

    def scan(self):
        """ Scan logic """
        num_drones = len(self.drone_list)
        num_waypoints = len(self.centres)

        waypoints_per_drone = num_waypoints // num_drones
        print(waypoints_per_drone)
        drone_waypoints = []
        start_idx = 0
        # Assign an equal number of waypoints to each drone
        for i in self.drone_list:
            end_idx = start_idx + waypoints_per_drone

            # For the last drone, assign any remaining waypoints
            if i.droneNumID == num_drones:
                end_idx = num_waypoints
                i.search_waypoints = self.centres[start_idx:end_idx]
                max_len = len(i.search_waypoints)
            else:
                i.search_waypoints = self.centres[start_idx:end_idx]
                start_idx = end_idx

        print(max_len)
        flag = False
        for t in range(max_len):
            if flag:
                print(t)
                break
            for drone in self.drone_list:
                result_check = self.read_fire_file()
                if result_check[0]:
                    print("fire", result_check[1])
                    self.target = result_check[1]
                    self.score  = result_check[2]
                    flag = True
                    break
                if t < len(drone.search_waypoints):
                    velocity = np.array(drone.search_waypoints[t] - drone.position)
                    drone.position_update_moveW(drone.search_waypoints[t])
                    self.PSOclient.moveByVelocityAsync(0, 0, -1, 1, drone.droneID)
                    time.sleep(1)
                    thread = self.PSOclient.moveToPositionAsync(drone.position, 10, self.angle_calc(velocity), drone.droneID)
                    time.sleep(1)
                    drone.pos_list.append(np.array(drone.position))
                    if drone.droneNumID == len(self.drone_list):
                        thread.join()


            print("drones moved by waypoint")


            # for drone in self.drone_list:
            #     image = self.PSOclient.takedroneimage(drone.droneNumID, drone.droneID)
            #     score = self.detect_fire(image)
            #     # cv2.imshow("output", image)
            #     # cv2.waitKey(0)
            #     # cv2.destroyAllWindows()
            #     if score > 0:
            #         print('fire_detected')
            #         self.show_fire(image)
            #         self.target = drone.position
            #         print(self.target)
            #         flag = True
            #         break

            self.explored_check()
            if len(self.centres) == 0:
                print("out of search pos")
                flag = True
                break

    def single_step(self):
        """ Step check """
        for i in self.centres:
            thread = self.PSOclient.moveToPositionAsync(i, 5, self.angle_calc([0,0,0]), "Drone1")
            thread.join()

    def zAxis_level(self):
        """ Assign z axis """
        for drone in self.drone_list:
            print(drone.position)
        max_height = float('-inf')
        min_height = float('inf')

        # Calculate the minimum and maximum height of the drones
        for drone in self.drone_list:
            height = drone.position[2]
            if height < min_height:
                min_height = height
            if height > max_height:
                max_height = height


        print(max_height, min_height)

        if max_height > -55:
            max_height = -55

        num_drones = len(self.drone_list)
        if num_drones > 1:
            spacing = 2

        copied_drone_list = list(self.drone_list)

        # Create an array between the max and min height
        z_array = []
        for i, drone in enumerate(copied_drone_list):
            drone_z = max_height - (i * spacing)
            z_array.append(drone_z)

        # Sort the copied_drone_list based on their current height in descending order
        copied_drone_list.sort(key=lambda drone: drone.position[2], reverse=True)

        # Assign drones to the closest z height in ascending order
        for drone in copied_drone_list:
            closest_height = min(z_array, key=lambda z: abs(z - drone.position[2]))
            print(closest_height)
            drone.position[2] = closest_height
            z_array.remove(closest_height)

            thread = self.PSOclient.moveToPositionAsync(drone.position, 2, self.angle_calc(drone.velocity), drone.droneID)
            time.sleep(0.1)
            thread.join()


    def drones_verify(self, drone):
        """ Verification check """
        num_steps = 4  # Number of 90-degree turns
        angle_step = 90  # Angle increment per step

        # orientation = self.PSOclient.getMultirotorState(drone.droneID)
        # print(orientation)
        initial_yaw = self.angle_calc(drone.velocity)
        for step in range(num_steps):
            print(step)
            # Calculate the target yaw angle
            target_yaw = initial_yaw + angle_step * (step + 1)

            # Turn the drone to the target yaw angle
            self.PSOclient.justturn(target_yaw, drone.droneID)
            time.sleep(1.2)

            image = self.PSOclient.takedroneimage(drone.droneNumID, drone.droneID)
            score = self.detect_fire(image)
            if score > 0:
                drone.fire_angle.append([initial_yaw,target_yaw])
                position = self.PSOclient.getMultirotorState(drone.droneID).kinematics_estimated.position
                x = position.x_val
                y = position.y_val
                z = position.z_val
                drone.fire_position = np.array([x, y, z])
                if score > drone.best_score_v:
                    drone.best_score_v = score
                    drone.best_im_v = image

                time.sleep(0.1)
                #self.show_fire(image)
            time.sleep(0.1)

    def drone_inspect_orbits(self):
        """ Orbit inscpection """
        #self.zAxis_level()
        best_drone_im = None
        best_score = 0
        for drone in self.drone_list:
            self.PSOclient.moveByVelocityAsync(0, 0, -0.5, 1, drone.droneID)
            time.sleep(0.1)
            self.drones_verify(drone)
            print(drone.droneID)
            if drone.best_score_v > best_score:
                best_score = drone.best_score_v
                best_drone_im = drone.best_im_v

        if best_drone_im is not None:
            self.show_fire(best_drone_im)

        self.fig_verify = plt.figure(figsize=(10, 10))
        self.ax_v = self.fig_verify.add_subplot()

        count_fire = 0

        for drone in self.drone_list:
            print(drone.fire_angle, drone.fire_position)

            drone_position = drone.fire_position

            if len(drone_position) != 0:
                count_fire += 1

            # Iterate over each set of angles and positions
            for i, angles in enumerate(drone.fire_angle):
                # Extract data for each angle set
                initial_yaw = angles[0]
                image_yaw = angles[1]


                # Convert yaw angles to radians
                initial_yaw_rad = np.deg2rad(initial_yaw)
                image_yaw_rad = np.deg2rad(image_yaw)

                # Compute the direction vectors for the yaw angles
                initial_direction = np.array([np.cos(initial_yaw_rad), np.sin(initial_yaw_rad)])
                image_direction = np.array([np.cos(image_yaw_rad), np.sin(image_yaw_rad)])

                # Plot the drone's location
                self.ax_v.plot(drone_position[0], drone_position[1], 'ro')

                # Plot the initial yaw angle line
                initial_line_start = np.array([drone_position[0], drone_position[1]])
                initial_line_end = initial_line_start + 25 * initial_direction
                self.ax_v.plot([initial_line_start[0], initial_line_end[0]], [initial_line_start[1], initial_line_end[1]], 'b--', label=f'Initial Yaw {i+1}')

                # Plot the image yaw angle line
                image_line_start = np.array([drone_position[0], drone_position[1]])
                image_line_end = image_line_start + 25 * image_direction
                self.ax_v.plot([image_line_start[0], image_line_end[0]], [image_line_start[1], image_line_end[1]], 'g--', label=f'Image Yaw {i+1}')

                cone_length = np.linalg.norm(image_line_end - image_line_start) + 50
                cone_angle = np.deg2rad(45)
                cone_left = image_line_start + cone_length * 0.5 * np.array([np.cos(image_yaw_rad + cone_angle), np.sin(image_yaw_rad + cone_angle)])
                cone_right = image_line_start + cone_length * 0.5 * np.array([np.cos(image_yaw_rad - cone_angle), np.sin(image_yaw_rad - cone_angle)])

                self.ax_v.fill([initial_line_start[0], cone_left[0], cone_right[0]], [initial_line_start[1], cone_left[1], cone_right[1]], 'r', alpha=0.3)
            # Set plot limits and labels
        self.ax_v.set_aspect('equal')
        self.ax_v.set_xlabel('X')
        self.ax_v.set_ylabel('Y')
        self.ax_v.set_title('Drone Location and Yaw Angles')


        return count_fire


    def logic_control(self):
        """ Logic fucntion """
        print("start logic")
        print(self.target)
        print(self.position_min, self.position_max)
        self.fig3d.scatter([pp[0] for pp in self.centres],
                     [pp[1] for pp in self.centres],
                    [-1*pp[2] for pp in self.centres], marker='^', c="b", zorder=0)

        true_target = [670,50.0]
        self.fig3d.scatter(true_target[0],true_target[1], self.heightmap[int(true_target[1]), int(true_target[0])], marker='^', s=150, c="g", label="True fire location")

        while len(self.centres) != 0:
            #self.scan()
            self.explore()
            print("end explore, investigating...")
            if self.target is not None:
                self.run()
                self.all_fire_locations  = self.read_all_fire_l()
                #self.zAxis_level()
                count_fire_value = self.drone_inspect_orbits()
                if count_fire_value > 5:
                    self.fig_verify.show()
                    break
                else:
                    self.target = None
                    with open(self.filename, 'w') as file:
                        file.truncate(0)
                    print(f"The contents of '{self.filename}' have been cleared.")

        for drone in self.drone_list:
            points = drone.pos_list
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            z = [p[2]*-1 + self.heightmap[int(p[0]), int(p[1])] if int(p[0]) in range(self.heightmap.shape[0]) and int(p[1]) in range(self.heightmap.shape[1]) else 45 for p in points] # check this works
            self.fig3d.plot(x, y, z, color='black')

        if self.target is not None:
            for i in self.all_fire_locations:
                self.fig3d.scatter(i[0], i[1], self.heightmap[int(i[1]), int(i[0])], marker='^', s=150, c="r", label="Fire predictions")
        self.fig3d.set_box_aspect([1., 1, 0.2])
        #self.fig3d.show()
        plt.show()


    def run(self):
        """ Convergemce logic """
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
            result_check = self.read_fire_file()
            if result_check[0]:
                print("fire", result_check[1])
                self.target = result_check[1]
                self.score  = result_check[2]
            print(t)
            if np.average([drone.fitness for drone in self.drone_list]) <= self.fitness_criterion:
                print("fitness converged ", t)
                break
            else:
                for drone in self.drone_list:
                    drone.fitness_calculation(self.target, self.search_radius)
                    time.sleep(0.1)
                    if drone.fitness <= self.fitness_criterion:
                        print(drone.droneID, " reached location")
                        thread = self.PSOclient.moveToPositionAsync(drone.position, 2.5, self.angle_calc(drone.velocity), drone.droneID)
                        time.sleep(0.1)
                        # thread.join()
                    else:
                        drone.velocity_update(self.drone_list, self.gbest_pos, self.PSOclient.getLidarData(drone.droneID, drone.droneNumID))
                        drone.position_update()
                        thread = self.PSOclient.moveToPositionAsync(drone.position, 2.5, self.angle_calc(drone.velocity), drone.droneID)
                        time.sleep(2)
                        #take image
                    # if drone.droneNumID == len(self.drone_list):

                    time.sleep(0.12)


                fitness = [drone.fitness for drone in self.drone_list]
                best_index = np.argmin(fitness)
                self.gbest_fitness = fitness[best_index]
                self.gbest_pos = self.drone_list[best_index].position


    def track_orbits(self, angle, drone):
        """ Track orbit """
        diff = abs(angle - self.start_angle)
        if self.start_flag == None:
            if (diff < 1):
                self.start_flag = True
                print("starting image")
                image = self.PSOclient.takedroneimage(drone.droneNumID, drone.droneID)
                score = self.detect_fire(image)
                if score > 0:
                    drone.fire_angle.append(angle)
                    drone.fire_position.append(self.PSOclient.getMultirotorState(drone.droneID).kinematics_estimated.position)
            return False
        else:
            if round(diff) >= 358:
                return True
            else:
                if round(diff) > self.prev_angle:
                    print(self.prev_angle)
                    image = self.PSOclient.takedroneimage(drone.droneNumID, drone.droneID)
                    score = self.detect_fire(image)
                    if score > 0:
                        drone.fire_angle.append(angle)
                        drone.fire_position.append(self.PSOclient.getMultirotorState(drone.droneID).kinematics_estimated.position)
                    self.prev_angle += 90
                    print(self.prev_angle)
                    print("1/4")

                return False

    def orbit(self, drone):
        """ Drone orbit """

        count = 0
        velocity = 2
        radius = 5

        center = [1,0]
        cx = float(center[0])
        cy = float(center[1])
        length = math.sqrt((cx*cx) + (cy*cy))
        cx /= length
        cy /= length
        cx *= radius
        cy *= radius

        pos_start = self.getCurrentPosition(drone.droneID)
        center = pos_start
        center.x_val += cx
        center.y_val += cy

        self.start_angle = -180
        self.start_flag = None
        self.prev_angle = 87
        while count < 1:
            # ramp up to full speed in smooth increments so we don't start too aggressively.
            lookahead_angle = velocity / radius
            # compute current angle
            pos = self.getCurrentPosition(drone.droneID)
            dx = pos.x_val - center.x_val
            dy = pos.y_val - center.y_val
            angle_to_center = math.atan2(dy, dx)

            camera_heading = (angle_to_center - math.pi) * 180 / math.pi

            # compute lookahead
            lookahead_x = center.x_val + radius * math.cos(angle_to_center + lookahead_angle)
            lookahead_y = center.y_val + radius * math.sin(angle_to_center + lookahead_angle)

            vx = lookahead_x - pos.x_val
            vy = lookahead_y - pos.y_val

            if self.track_orbits(angle_to_center * 180 / math.pi, drone):
                count += 1
                print("completed {} orbits".format(count))

            self.PSOclient.moveByVelocityZAsync(vx, vy, drone.position[2], 2, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, camera_heading), drone.droneID)

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
