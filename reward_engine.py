import numpy as np
import cv2
import math
import json
import io
import ast
import tempfile
import os
import zipfile
import time
from numpy.fft import fft2, ifft2, fftshift


def distance(p1, p2):
    tmp_v = (p1[0]-p2[0], p1[1]-p2[1])
    return math.sqrt(tmp_v[0]**2+ tmp_v[1]**2)

def sum_coor(c1, c2):
    return(c1[0] + c2[0], c1[1] + c2[1])

def load_all_at_once(npz_file, zip_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_file.extract("dabug_inputs.npz", tmpdir)
        extracted_npz_path = os.path.join(tmpdir, "dabug_inputs.npz")

        with np.load(extracted_npz_path) as npz:
            data = [npz[name] for name in npz.files]
            return data
        
class Map(str):
    canvas_counter = 0
    def __init__(self, config):
                
        self.config = config
        self.dim = config['initial_map_resolution']
        self.canvas = np.zeros(self.dim, dtype=float)
        self.discovery_mask = np.zeros(self.dim).astype(bool)
        self.center = (self.dim[0]//2, self.dim[1]//2)
        self.last_coors = self.center
        self.frame_res = self.config['input_resolution']
        # self.margin = self.config['frame_comparision_margin']
        self.index = Map.canvas_counter
        self.name = "Canvas_map_" + str(self.index)
        self.extend_offset = (0,0)
        self.character_mask = np.ones(self.frame_res , dtype=float)
        self.map_distance = 0
        Map.canvas_counter += 1

        self.character_mask[3*16-4:4*16, 2*16+8: 3*16+8] = 255.0
    
    def __hash__(self):
        return hash(self.name)

    def efficient_save(self, zip_file): #Could fail and not close the file, problematic but ok for now

        complete_path = 'saved_' + self.name

        discovery_mask_buffer = io.BytesIO()
        canvas_buffer = io.BytesIO()
        
        # Save the array to this buffer using numpy.save
        np.save(discovery_mask_buffer, self.discovery_mask)
        np.save(canvas_buffer, self.canvas)
        
        # Move the buffer cursor to the start
        discovery_mask_buffer.seek(0)
        canvas_buffer.seek(0)
        
        # Add the serialized NumPy array to the ZIP file
        zip_file.writestr(complete_path + '_discovery_mask.npy', discovery_mask_buffer.getvalue())
        zip_file.writestr(complete_path + '_canvas.npy', canvas_buffer.getvalue())
        
        map_stats_dict = {"dim": [self.dim[0], self.dim[1]],
                          "last_coors": [self.last_coors[0], self.last_coors[1]],
                          "index": self.index,
                          "name":self.name,
                          "distance":self.map_distance,
                          "extend_offset": list(self.extend_offset)}
        map_stats_json = json.dumps(map_stats_dict, ensure_ascii=False)
        zip_file.writestr(complete_path + "_stats.json", map_stats_json)


        # self.discovery_mask.tofile(complete_path + '_discovery_mask.npy')
        # self.canvas.tofile(complete_path + '_canvas.npy')
    
    def efficient_load(self, zip_file, name):
        complete_path = 'saved_' + name
        
        self.name = name
        # Load the discovery_mask NumPy array
        with zip_file.open(complete_path + '_discovery_mask.npy') as f:
            discovery_mask_buffer = io.BytesIO(f.read())
            discovery_mask_buffer.seek(0)
            self.discovery_mask = np.load(discovery_mask_buffer)

        # Load the canvas NumPy array
        with zip_file.open(complete_path + '_canvas.npy') as f:
            canvas_buffer = io.BytesIO(f.read())
            canvas_buffer.seek(0)
            self.canvas = np.load(canvas_buffer)
            
        with zip_file.open(complete_path +'_stats.json') as f:
            map_stats_json = json.load(f)
        
        self.dim = (int(map_stats_json["dim"][0]), int(map_stats_json["dim"][1]))
        self.last_coors = (int(map_stats_json["last_coors"][0]), int(map_stats_json["last_coors"][1]))
        self.extend_offset = (int(map_stats_json["extend_offset"][0]), int(map_stats_json["extend_offset"][1]))
        self.index = int(map_stats_json["index"])
        self.map_distance = float(map_stats_json["distance"])

        # print(f"Loaded {complete_path}_discovery_mask.npy and {complete_path}_canvas.npy")
        

    def show(self):
        # Grid
        cropped = self.crop()
        cv2.imshow(self.name, (cropped/255))
        cv2.moveWindow(self.name, self.frame_res[1] * (self.index%5), self.frame_res[0] * (self.index//5))
        cv2.waitKey(1)
    
    def close(self):
        cv2.destroyWindow(self.name)

    def start(self, frame):
        return self.append(frame, (0,0))
    
    def save(self, path):
        cropped = self.crop()
        cv2.imwrite(path + 'saved_' + self.name + '.png',cropped)
    
    def crop(self):
        y_nonzero, x_nonzero = np.nonzero(self.discovery_mask)
        if len(x_nonzero) == 0 or len(y_nonzero) == 0: return None
        y = np.min(y_nonzero)
        h = np.max(y_nonzero) - y
        x = np.min(x_nonzero)
        w = np.max(x_nonzero) - x
        crop = self.canvas[y:y+h,x:x+w]
        return crop
        
    def extend(self, direction):
        
        chunk = self.config['chunk_extension_size']

        if direction == 0:
            self.canvas =           np.pad(self.canvas, ((chunk,0),(0,0)), constant_values=0)
            self.discovery_mask =   np.pad(self.discovery_mask, ((chunk,0),(0,0)), constant_values=False)
            self.extend_offset = (self.extend_offset[0] + chunk, self.extend_offset[1])
            self.dim = (self.dim[0] + chunk, self.dim[1])
        elif direction == 1:
            self.canvas =           np.pad(self.canvas, ((0,chunk),(0,0)), constant_values=0)
            self.discovery_mask =   np.pad(self.discovery_mask, ((0,chunk),(0,0)), constant_values=False)
            self.dim = (self.dim[0] + chunk, self.dim[1])
        elif direction == 2:
            self.canvas =           np.pad(self.canvas, ((0,0), (chunk,0)), constant_values=0)
            self.discovery_mask =   np.pad(self.discovery_mask, ((0,0), (chunk,0)), constant_values=False)
            self.extend_offset = (self.extend_offset[0], self.extend_offset[1] + chunk)
            self.dim = (self.dim[0], self.dim[1] + chunk)
        else:
            self.canvas =           np.pad(self.canvas, ((0,0), (0,chunk)), constant_values=0)
            self.discovery_mask =   np.pad(self.discovery_mask, ((0,0), (0,chunk)), constant_values=False)
            self.dim = (self.dim[0], self.dim[1] + chunk)


    def calculate_index(self, i, offset = False, margin = False, custom_coords = False, square = False, extended = True):
        result = 0
        axis = 0
        if not offset: offset = (0,0)

        if i == 0 or i == 1:
            axis = 0
        elif i == 2 or i == 3:
            axis = 1
        
        # if square: 
        #     if i == 2: result += 8
        #     if i == 3: result -= 8
        
        if extended: 
            if i == 0 or i == 1: result += self.extend_offset[0]
            if i == 2 or i == 3: result += self.extend_offset[1]

        if not custom_coords: result += self.last_coors[axis]
        else: result += custom_coords[axis]
        
        result += offset[axis]

        # if offset[(axis+1)%2] != 0:
        # if (i == 0 or i == 2) and margin: 
        #     result += self.margin * abs(offset[(axis)%2]//16)
        # if (i == 1 or i == 3) and margin:             
        #     result -= self.margin * abs(offset[(axis)%2]//16)  

        if i == 1 or i == 3: 
            result += self.frame_res[axis]
        #     result -= self.margin * margin
            

        return result

    def compare_with_offset(self, arr1, arr2, offset_x, offset_y):

        h1, w1 = arr1.shape
        h2, w2 = arr2.shape

        x1_min = max(0, offset_x)
        y1_min = max(0, offset_y)
        x1_max = min(w1, w2 + offset_x)
        y1_max = min(h1, h2 + offset_y)
        
        x2_min = max(0, -offset_x)
        y2_min = max(0, -offset_y)
        x2_max = min(w2, w1 - offset_x)
        y2_max = min(h2, h1 - offset_y)

        if x1_min < x1_max and y1_min < y1_max:
            region1 = arr1[y1_min:y1_max, x1_min:x1_max]
            region2 = arr2[y2_min:y2_max, x2_min:x2_max]
            comparison = abs(region1 - region2)
            
            return comparison
        else:
            return None 
        
    def get_current(self):
        return self.canvas[    self.calculate_index(0):self.calculate_index(1), self.calculate_index(2):self.calculate_index(3)]
    
    def compare(self, frame, offset=(0,0), coords=False, debug=False): # 8 more at the sides so it is square, otherwise the comparision would be diferent when moving vertically and horizontally
        
        if self.calculate_index(0, offset, False, coords) <= 0:
            self.extend(0)
        elif self.calculate_index(1, offset, False, coords) >= self.dim[0]:
            self.extend(1)
        elif self.calculate_index(2, offset, False, coords) <= 0:
            self.extend(2)
        elif self.calculate_index(3, offset, False, coords) >= self.dim[1]:
            self.extend(3)

        if not coords: 
            coords = self.last_coors
            
        a = self.canvas[    self.calculate_index(0, False, True, coords):self.calculate_index(1, False, True, coords), \
                            self.calculate_index(2, False, True, coords):self.calculate_index(3, False, True, coords)]

        score = 0
        try:

            dif_img =  self.compare_with_offset(a, frame, offset[1], offset[0])
            #  dif_img[-16:, :] = 0.0 # in case you want to trim a portion of the screen 
            score = (dif_img/255).sum()/dif_img.size #/c
        except:
            #This should basically never happen, but somehow frame can be null with long runs 
            #I can't easilly debug an error that happens 20+ hours in,
            #So for now this 
            print("ERROR") 
        if False:
            print(score)
            cv2.imshow("a", a)
            cv2.moveWindow("a", 500,0)
            cv2.imshow("frame", frame)
            cv2.moveWindow("frame", 700,0)
            cv2.imshow("dif_img", dif_img)
            cv2.moveWindow("dif_img", 900,0)
            cv2.waitKey(2000)

        return score
        
    def append(self, frame, offset):


        score =  np.count_nonzero(~self.discovery_mask[ self.calculate_index(0, offset) : self.calculate_index(1, offset), \
                                                        self.calculate_index(2, offset) : self.calculate_index(3, offset)]) / (self.frame_res[0]*(self.frame_res[1]))
        
        if score > 0 or True:
        
            self.canvas[                        self.calculate_index(0, offset) : self.calculate_index(1, offset), \
                                                self.calculate_index(2, offset) : self.calculate_index(3, offset)] \
                            = frame
            
            self.discovery_mask[                self.calculate_index(0, offset) : self.calculate_index(1, offset), \
                                                self.calculate_index(2, offset) : self.calculate_index(3, offset)] \
                            = True
                
        self.last_coors = (self.last_coors[0] + offset[0], self.last_coors[1] + offset[1])

        if self.config['use_distance']:
            d = distance(self.last_coors, self.center)
            k = self.config['distance_k']
            score = score * k * (d + self.map_distance)
        return score 

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.__str__()

class Engine:
    
    
    def __init__(self, config):

        self.config = config
        self.map_archive = {}
        self.cm = None
        self.tick = 0
        self.repeat_ward = []
        self.deleted_maps = []
        self.past_entry_points = []
        self.initial_frame = None

        self.just_new = False
        self.preamble_size = self.config['preamble_size']
        self.previous_data = [np.zeros(config['input_resolution'])] * self.preamble_size
    

        self.debug_data = {
            "blanks" : 0,
            "repeats" : 0,
            "surround" : 0,
            "direct_ep" : 0,
            "adjacent_ep" : 0,
            "surround_ep" : 0,
            "tree" : 0,
            "new" : 0
        }
        
    def save(self,path):
        for m in self.map_archive.keys():
            m.save(path)

        if len(self.previous_data) == 0: return
        height, width = self.previous_data[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # You can use 'XVID' for .avi format
        out = cv2.VideoWriter(path + "_anim.avi", fourcc, 15, (width, height))
        for img in self.previous_data:
            out.write(cv2.cvtColor(np.uint8(img), cv2.COLOR_GRAY2BGR))  # Convert to BGR format for OpenCV
        out.release()
    
    def map_while(self, fun, vals , threshold):
        for i, val in enumerate(vals):
            a = fun(val)
            if a < threshold:
                return i, True
        return 0, False
        
    def detect_significant_minimum(self, values, threshold): # pokemon 0.04
        
        return list(values).index(min(values)), any([(val) < threshold for val in values])
    
    
    def append(self, frame, action):

        try:
    
            self.previous_data.append(frame)

            if np.all(frame==0) or np.all(frame==255): 
                self.debug_data["blanks"] += 1
                return 0 #not even one pixel, skip

            if len(self.repeat_ward) > self.config['repeat_ward_max_size']: self.repeat_ward.pop(0)
            repeat_mult = float( bool( not any(np.array_equal(frame, x) for x in self.repeat_ward)))
            if repeat_mult != 0: self.debug_data["repeats"] += 1
            self.repeat_ward.append(frame)
            
            #if first frame, create map
            if len(self.map_archive) == 0:
                tmp_map = Map(self.config)
                score = tmp_map.start(frame)
                self.map_archive[tmp_map] = {tmp_map.last_coors: [[tmp_map, tmp_map.last_coors, 0],]} # map (at) -> coords (leads to) -> [ map , (at) coords , (last seen) tick ]
                self.past_entry_points.append((tmp_map, tmp_map.last_coors))
                if len(self.past_entry_points) > self.config['past_entry_point_list_max_size']: self.past_entry_points.pop(1)
                self.cm = tmp_map
                self.initial_frame = (self.cm, self.cm.last_coors)
                self.debug_data["new"] += 1
                return score * repeat_mult

            # check same map, same place and surroundings
            offsets = [[(i,0),(-i,0),(0,-i),(0,i)] for i in self.config['inmediate_offsets']]# + [[(i,0),(-i,0),(0,i),(0,-i)] for i in range(14*2, 18*2)]
            offsets = [(0,0)] + [x for xs in offsets for x in xs] # 0,0 is first because if all of them are equal, the first is returned by argmin
            comparations = np.array(list(map(lambda x: self.cm.compare(frame, x), offsets)))
            best, found = self.detect_significant_minimum(comparations, self.config['inmediate_threshold'])

            if found: #if most of them are valid, dont pick 
                indices_ordenados = np.argsort(comparations)
                second_best= indices_ordenados[1]

                if action > 3: action = 0
                else: action += 1 

                if abs(comparations[best] - comparations[second_best]) < self.config['stay_threshold_margin'] and second_best == action:
                    best = action # If the best is close to staying in place, stay in place

                score = self.cm.append(frame, offsets[best])
                self.debug_data["surround"] += 1
                return score * repeat_mult
            

            # check entry points
            if self.cm.last_coors in self.map_archive[self.cm]:
                # comparation = self.map_archive[self.cm][self.cm.last_coors][0].compare(frame, (0,0))
                comparations = np.array(list(map(lambda x: x[0].compare(frame, coords=x[1], debug=False), self.map_archive[self.cm][self.cm.last_coors])))
                best, found = self.detect_significant_minimum(comparations, self.config['inmediate_entry_point_threshold'])
                if found: #if most of them are valid, dont pick 
                    # print("Appending ar entry point")
                    pre_map = self.cm
                    self.cm = self.map_archive[self.cm][self.cm.last_coors][best][0]
                    self.map_archive[pre_map][pre_map.last_coors][best][2] = self.tick
                    self.cm.last_coors = self.map_archive[pre_map][pre_map.last_coors][best][1]
                    score = self.cm.append(frame, (0,0))
                    self.debug_data["direct_ep"] += 1
                    return score * repeat_mult
            else:
                #There might be an entry point adyacent to the current one
                close = self.map_archive[self.cm].keys()
                close = list(filter(lambda x: distance(self.cm.last_coors, x) <= 32, close))
                # For now, lets get the first one
                if len(close) > 0:
                    closes_coor = close[0]
                    comparations = np.array(list(map(lambda x: x[0].compare(frame, coords=x[1], debug=False), self.map_archive[self.cm][closes_coor])))
                    best, found = self.detect_significant_minimum(comparations,self.config['close_entry_point_threshold'])

                    if found: 
                        pre_map = self.cm
                        self.map_archive.setdefault(self.map_archive[self.cm][closes_coor][best][0], {})
                        self.map_archive[self.cm][self.cm.last_coors] = []
                        self.map_archive[self.cm][self.cm.last_coors].append([self.map_archive[self.cm][closes_coor][best][0], self.map_archive[self.cm][closes_coor][best][1], self.tick])
                        self.cm = self.map_archive[self.cm][self.cm.last_coors][0][0] #it's the first one
                        self.map_archive[pre_map][pre_map.last_coors][0][2] = self.tick
                        self.cm.last_coors = self.map_archive[pre_map][pre_map.last_coors][0][1]
                        score = self.cm.append(frame, (0,0))
                        self.past_entry_points.append((pre_map, pre_map.last_coors))
                        if len(self.past_entry_points) > self.config['past_entry_point_list_max_size']: self.past_entry_points.pop(1)
                        self.debug_data["adjacent_ep"] += 1
                        return score * repeat_mult


            # Entry point not found, last chance to find it with offset due to desync of animations       
            search = []
            for index, m in enumerate(self.past_entry_points):
                for i in self.config['close_entry_point_offsets']:
                    search.append((m[0], m[1], (0,0), index)) # Would not really be necessary but for now, otherwise will offset black screens
                    search.append((m[0], m[1], (i,0), index))
                    search.append((m[0], m[1], (-i,0), index))
                    search.append((m[0], m[1], (0,i), index))
                    search.append((m[0], m[1], (0,-i), index))
                
            reversed_search = list(reversed(search))
            best, found = self.map_while(lambda x: x[0].compare(frame, coords=x[1], offset=x[2] , debug=False), reversed_search, threshold=self.config['past_entry_point_threshold'])
            if found: #if most of them are valid, dont pick 
                # print("Appending ar entry point")
                pre_map = self.cm
                self.cm = reversed_search[best][0] #self.map_archive[self.cm][self.cm.last_coors][best][0]
                self.cm.last_coors = sum_coor(reversed_search[best][2], reversed_search[best][1]) #self.map_archive[pre_map][pre_map.last_coors][best][1]
                self.map_archive[pre_map].setdefault(pre_map.last_coors, [])
                self.map_archive[pre_map][pre_map.last_coors].append([self.cm, self.cm.last_coors, self.tick]) #[best][2] = self.tick

                score = self.cm.append(frame, (0,0))
                #self.past_entry_points.append((pre_map, pre_map.last_coors))
                # print("whole:_", self.past_entry_points)
                # print("deleted:_", self.past_entry_points[reversed_search[best][3]+1:])
                # del self.past_entry_points[reversed_search[best][3]+1:]
                self.past_entry_points.append((pre_map, pre_map.last_coors))
                if len(self.past_entry_points) > self.config['past_entry_point_list_max_size']: self.past_entry_points.pop(1)
                self.debug_data["surround_ep"] += 1
                return score * repeat_mult
            
            
            
            """checking_map = self.cm
            checking_coords = self.cm.last_coors
            visited = set([checking_map])
            queue = [(checking_map, checking_coords)]
            while len(queue) != 0:
                next = queue.pop(0)
                checking_map = next[0]
                checking_coords = next[1]
                visited.add(checking_map)
                #if not (checking_coords[0] == 200 and checking_coords[1] == 200): break

                search = [(0,0),(16,0),(-16,0),(0,16),(0,-16),]
                best, found = self.map_while(lambda x: checking_map.compare(frame, coords=checking_coords, offset=x , debug=False), search, threshold=0.04)

                if found:
                    pre_map = self.cm
                    self.cm = checking_map #self.map_archive[self.cm][self.cm.last_coors][best][0]
                    self.cm.last_coors = sum_coor(search[best], checking_coords) #self.map_archive[pre_map][pre_map.last_coors][best][1]
                    self.map_archive[pre_map].setdefault(pre_map.last_coors, [])
                    self.map_archive[pre_map][pre_map.last_coors].append([self.cm, self.cm.last_coors, self.tick]) #[best][2] = self.tick

                    score = self.cm.append(frame, (0,0))

                    self.past_entry_points.append((pre_map, pre_map.last_coors))
                    if len(self.past_entry_points) > 100: self.past_entry_points.pop(1)
                    self.debug_data["tree"] += 1
                    return score * repeat_mult
                else:
                    for c in search:
                        if sum_coor(c, tuple(checking_coords)) not in self.map_archive[checking_map]: continue
                        # if not (checking_coords[0] == 200 and checking_coords[1] == 200): continue
                        leads_to = self.map_archive[checking_map][sum_coor(c, tuple(checking_coords))]
                        leads_to = list(filter(lambda x:  x[0] not in visited, leads_to))
                        if len(leads_to) == 0: continue
                    
                        queue = queue + leads_to


            if self.initial_frame[0].compare(frame, coords=self.initial_frame[1]) < 0.04: #if most of them are valid, dont pick 

                self.cm = self.initial_frame[0]
                self.cm.last_coors = self.initial_frame[1]
                return -2"""

            # CREATING NEW MAP
            tmp_map = Map(self.config)
            self.map_archive[tmp_map] = {}
            tmp_map.map_distance += self.cm.map_distance + distance(self.cm.last_coors, self.cm.center)
            score = tmp_map.start(frame)

            self.map_archive.setdefault(self.cm, {} )
            self.map_archive[self.cm].setdefault(self.cm.last_coors, [] )
            self.map_archive[self.cm][self.cm.last_coors].append([tmp_map, tmp_map.last_coors, self.tick]) # map (at) -> coords (leads to) -> map , (at) coords , (last seen) tick
        
            self.map_archive.setdefault(tmp_map, {tmp_map.last_coors: [] })
            self.map_archive[tmp_map].setdefault(tmp_map.last_coors, [])
            self.map_archive[tmp_map][tmp_map.last_coors].append([self.cm, self.cm.last_coors, self.tick]) # map (at) -> coords (leads to) -> map , (at) coords , (last seen) tick

            self.past_entry_points.append((self.cm, self.cm.last_coors))
            if len(self.past_entry_points) > self.config['past_entry_point_list_max_size']: self.past_entry_points.pop(1)
            self.cm = tmp_map
            self.debug_data["new"] += 1
            return score * self.config['new_map_multiplier'] * repeat_mult
        except Exception as e:
            
            print(e)
            return 0

    def show(self):
        self.cm.show()

    def close(self):
        for a,b in self.map_archive:
            b.close()
    
    def efficient_save(self, zip_file):

        names = list(set([m[1].name for m in self.map_archive]))
        archive = [ [m[0],m[1].name,m[2], m[3].name] for m in self.map_archive ]
        engine_dict = {'ci' : self.cm, 'tick': self.tick, "map_names": names, "archive": archive}
        engine_dict_json = json.dumps(engine_dict, ensure_ascii=False)
        zip_file.writestr('engine_stats.json', engine_dict_json)

        done = set()
        for m in self.map_archive:
            if not m[1] in done:
                m[1].efficient_save(zip_file)
                done.add(m[1])

    def efficient_load(self, zip_file):
        
        with zip_file.open('engine_stats.json') as f:
            engine_dict = json.load(f)
        
        self.cm = int(engine_dict["ci"])
        self.tick = int(engine_dict["tick"])
        names = engine_dict["map_names"]
        maps = [Map(self.config) for name in names]
        list(map(lambda x: x[0].efficient_load(zip_file, x[1]), zip(maps, names)))
        maps = {x[1]:x[0] for x in zip(maps, names)}
        
        archive = engine_dict["archive"]
        self.map_archive = [[list(x[0]), maps[x[1]], x[2], maps[x[3]]] if x[3] in names else [list(x[0]), maps[x[1]], x[2], maps[x[1]]] for x in archive]

    def efficient_dict_save(self, zip_file):
    
        names = list(set([m.name for m in self.map_archive.keys()]))

        serializable_archive = {map_obj.name: 
                                                {str(coord_key): 
                                                            [[entry_point[0].name, entry_point[1], entry_point[2]]
                                                            for entry_point in entry_point_list]
                                                for coord_key, entry_point_list in map_value.items()}
                                for map_obj, map_value in self.map_archive.items()}

        eps = [[x[0].name, x[1]] for x in self.past_entry_points]
        ini_frame = (self.initial_frame[0].name, self.initial_frame[1])
        engine_dict = {'ci' : self.cm.name, 'tick': self.tick, "map_names": names, "archive": serializable_archive, "past_eps": eps,  "initial_frame": ini_frame}

        engine_dict_json = json.dumps(engine_dict, ensure_ascii=False)
        zip_file.writestr('engine_stats.json', engine_dict_json)

        
        
        start = time.time()
        with zip_file.open('dabug_inputs.npz', "w") as f:
            
            preamble = self.previous_data
            # print(len(preamble))
            if len(preamble) < self.preamble_size:
                preamble = [np.zeros(self.config['input_resolution'])] * (self.preamble_size - len(preamble)) + preamble
            elif len(preamble) > self.preamble_size:
                preamble = preamble[-self.preamble_size:]
            # print(len(preamble))

            np.savez_compressed(f, *preamble)
        end = time.time()
        # print(f"Saved {len(preamble)} in {end-start} seconds")

        done = set()
        for m in self.map_archive:
            if not m in done:
                # print(m)
                m.efficient_save(zip_file)
                done.add(m)

    def efficient_dict_load(self, zip_file):
            
        with zip_file.open('engine_stats.json') as f:
            engine_dict = json.load(f)
        
        start = time.time()
        self.previous_data = load_all_at_once("dabug_inputs.npz", zip_file)
        end = time.time()
        # print(f"Loaded {len(self.previous_data)} in {end-start} seconds")
        if len(self.previous_data) < self.preamble_size:
                self.previous_data = [np.zeros(self.config['input_resolution'])] * (self.preamble_size - len(self.previous_data)) + self.previous_data
        # print(len(self.previous_data ), "DATOS CARGADOS")


        names = engine_dict["map_names"]
        maps = [Map(self.config) for name in names]
        list(map(lambda x: x[0].efficient_load(zip_file, x[1]), zip(maps, names)))
        maps = {x[1]:x[0] for x in zip(maps, names)}
        
        self.cm = maps[engine_dict["ci"]]
        self.tick = int(engine_dict["tick"])
        self.initial_frame = (maps[engine_dict["initial_frame"][0]], tuple(engine_dict["initial_frame"][1]))

        self.past_entry_points = [(maps[x[0]], x[1]) for x in engine_dict["past_eps"]]
        archive = engine_dict["archive"]
        self.map_archive = {maps[map_name]: 
                                    {ast.literal_eval(coord_key): 
                                                [[maps[entry_point[0]], entry_point[1], entry_point[2]]
                                                for entry_point in entry_point_list]
                                    for coord_key, entry_point_list in map_value.items()}
                            for map_name, map_value in archive.items()}