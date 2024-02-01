import airsim
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


    def takedroneimage(self, numId, droneID):
        responses = self.client.simGetImages([
        airsim.ImageRequest("f"+str(numId), airsim.ImageType.Scene)], droneID)
        #airsim.write_file(os.path.normpath('./images/fire_check_'+str(droneID)+'.png'), response.image_data_uint8)
        img = cv2.imdecode(np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img

    def getMultirotorState(self, id):
        return self.client.getMultirotorState(vehicle_name = id)


class droneClient(object):
    """docstring for Drone."""

    def __init__(self, id, num):
        self.droneID = id
        self.droneNumID = num
        self.score = 0
        self.position = [0,0,0]
        self.image = None



class FirePSO(object):
    """docstring for PSO."""
    def __init__(self):
        super(FirePSO, self).__init__()
        self.drone_list = []
        self.PSOclient = Client()
        self.start()
        #self.test()


    def write_to_file(self, array, score):
        filename = "Fire_location.txt"
        with open(filename, 'w') as file:
            line = ' '.join(str(num) for num in array) + ' | ' + str(score)
            file.write(line)


    def write_to_file_simple(self, array):
        filename = "Fire_l_records.txt"
        with open(filename, 'a') as file:
            line = '[ ' + ' '.join(str(num) for num in array) + ' ]\n'
            file.write(line)


    def start(self):
        #self.client.confirmConnection()
        id = "Drone"
        for i in range(20):
            id = "Drone" + str(i+1)
            print(id)
            newDrone = droneClient(id, i+1)
            self.drone_list.append(newDrone)

        lowest_score = 0.0

        flag = True
        while flag:
            best_drone = self.drone_list[0]
            for drone in self.drone_list:
                state_val = self.PSOclient.getMultirotorState(drone.droneID)
                current_position = state_val.kinematics_estimated.position
                time.sleep(0.05)
                drone.position = np.array([current_position.x_val, current_position.y_val, current_position.z_val])
                image = self.PSOclient.takedroneimage(drone.droneNumID, drone.droneID)
                drone.image = image

                #### if score > 0 then write to fire file anyway
                drone.score = self.test_detect(image)
                if drone.score > 0:
                    self.write_to_file_simple(drone.position)
                if drone.score > lowest_score:
                    if drone.score > best_drone.score:
                        best_drone = drone
                    print(drone.score)
                    time.sleep(0.15)


            print(best_drone, best_drone.score)


            if best_drone.score > lowest_score:
                print('fire_detected')
                # try:
                #self.show_fire(best_drone.image)
                # except:
                #     print("wierd error")
                self.target = best_drone.position
                print(self.target)
                self.write_to_file(self.target, best_drone.score)
                lowest_score = best_drone.score
                print(lowest_score)

            for i in self.drone_list:
                i.score = 0

            time.sleep(3)

    def test_detect(self, image):
        # cv2.imshow("ds", image)
        # cv2.waitKey(0)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


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
        #
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

if __name__ == '__main__':
    FirePSO()
