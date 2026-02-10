import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import simpledialog
import matplotlib.patches as patches
import time
from colorama import Fore, Style, init
import torch

device = torch.device("cuda")  # assume GPU per your request

print("")
print(Fore.BLUE + "Input the 2D CAD model" + Style.RESET_ALL)

# Data storage
vertex = []
line = []  # Make sure this is defined globally
circle = []
boundary_coordinates_1 = []
horizontal = []
vertical = []
brk_id = []
brk_id_circle =[]
# Setup figure
fig, ax = plt.subplots()
ax.set_title("Draw polygons (1), circles (2), arcs (3). Press 'q' to quit.")
plt.axis('equal')
plt.grid(True)

# State variables
current_mode = 'polygon'  # 'polygon', 'circle', or 'arc'
current_polygon = []
drawing_polygon = True

# Tkinter root (for dialogs)
root = tk.Tk()
root.withdraw()

def autoscale():
    ax.relim()
    ax.autoscale_view()
    ax.set_aspect('equal', adjustable='datalim')
    fig.canvas.draw()

def draw_polygon(points, style='b-'):
    xs, ys = zip(*points)
    ax.plot(xs, ys, style)
    autoscale()

def draw_circle(center, radius):
    circle = patches.Circle(center, radius, edgecolor='green', facecolor='none', linestyle='--')
    ax.add_patch(circle)
    autoscale()

def draw_arc(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    mid = (p1 + p2) / 2
    radius = np.linalg.norm(p1 - p2) / 2
    theta1 = np.degrees(np.arctan2(p1[1] - mid[1], p1[0] - mid[0]))
    theta2 = np.degrees(np.arctan2(p2[1] - mid[1], p2[0] - mid[0]))
    arc = patches.Arc(mid, 2 * radius, 2 * radius, angle=0, theta1=theta1, theta2=theta2, color='purple')
    ax.add_patch(arc)
    autoscale()

def onkey(event):
    global current_polygon, vertex, current_mode

    if event.key == 'q':
        if current_polygon:
            vertex.append(('polygon', current_polygon.copy()))
        print("Final data:\n", vertex)
        plt.close()

    elif event.key == 'c' and current_mode == 'polygon':
        if len(current_polygon) > 2:
            xs, ys = zip(*[current_polygon[-1], current_polygon[0]])
            ax.plot(xs, ys, 'b--')
            vertex.append(('polygon', current_polygon.copy()))
            current_polygon = []
            autoscale()

    elif event.key == 'escape':
        current_polygon = []
        fig.canvas.draw()

    elif event.key == '1':
        current_mode = 'polygon'
        print("Switched to polygon mode.")

    elif event.key == '2':
        current_mode = 'circle'
        try:
            input_str = simpledialog.askstring("Input Circle", "Enter center_x,center_y,radius (or 'break'):")
            parts = input_str.strip().split()

            circles = []
            for p in parts:
                if p.lower() == "break":
                    circles.append("BREAK")
                    brk_id_circle.append(len(circles) - 1)
                    break  # optional: stop further input
                else:
                    cx, cy, r = map(float, p.strip().split(','))
                    circles.append((cx, cy, r))

            # Clean circles (no BREAK)
            clean_circles = [c for c in circles if c != "BREAK"]

            # Draw them
            for (cx, cy, r) in clean_circles:
                draw_circle((cx, cy), r)

            # Save raw + clean data
            circle.extend(circles)  
            vertex.append(('circle', clean_circles))
            print("Circle data updated:", circle)
            print("BREAK index positions:", brk_id)

        except Exception as e:
            print("Invalid input for circle:", e)

    elif event.key == 'i':
        try:
            input_str = simpledialog.askstring("Manual Input", "Enter points (e.g., 0,0 1,0 1,1 0,1 break):")
            parts = input_str.strip().split()

            pts = []
            

            for i, p in enumerate(parts):
                if p.lower() == "break":
                    pts.append("BREAK")
                    brk_id.append(len(pts) - 1)  # Position of "BREAK" in pts
                    break  # Optional: stop input here
                else:
                    x, y = map(float, p.strip().split(','))
                    pts.append((x, y))

            # Only draw valid points, ignore "BREAK"
            clean_pts = [p for p in pts if p != "BREAK"]
            draw_polygon(clean_pts, style='g-')

            line.extend(pts)  # Keep raw data, including "BREAK"
            vertex.append(('polygon', clean_pts))  # Store clean version
            print("Line data updated:", line)
            print("BREAK index positions:", brk_id)

        except Exception as e:
            print("Invalid input for manual polygon:", e)
# Connect events

fig.canvas.mpl_connect('key_press_event', onkey)

plt.show()
#-------------------------------------------------------------------------Meshing begins-------------------------------------------------#
print("Finding Xmax & Ymax...")
# Filter only (x, y) points
points = [p for p in line if isinstance(p, tuple)]

# Extract x and y separately
x_values = [p[0] for p in points]
y_values = [p[1] for p in points]

# Compute max values
x_max = max(x_values)
y_max = max(y_values)

print("Xmax =", x_max)
print("Ymax =", y_max)
#-----------------------------------------------------------#

# def find_repeated():

# Calcuation of mid-boundary coordinates
# Note: We must try to avoid giving geometries smaller than mesh element size
# Example: We want a grid of 129x129 (includes boundaries) so we must input del_h = 10/(129-2) = 10/(127)

del_h = 0.126             #0.0787              
tol = 8
tolerance = 1e-8
space_laps = 1e-4    # (set your input precision)

cad_st = time.time()
# calculation for line
if (len(line)!=0):
    for i in range(0, len(line)-1, 1):
        X1, X2 = line[i][0], line[i+1][0]
        Y1, Y2 = line[i][1], line[i+1][1]
        
        
        if ((X1 != "B" or Y1 !="R") and (X2 != "B" or Y2 !="R") ):
            
            
            if (abs(X2 - X1) < 1e-5):  # consider vertical line
                # Use np.arange on Y, keep X constant        
                if (Y2 > Y1):
                    for y in np.arange(Y1, Y2, space_laps):
                        boundary_coordinates_1.append((X1, y))
                        vertical.append((X1,y))
                elif (Y2 < Y1):
                    for y in np.arange(Y1, Y2, -(space_laps)):
                        boundary_coordinates_1.append((X1, y))
                        vertical.append((X1,y))


            elif (abs(Y2-Y1) < 1e-5):     # consider horizontal line (simply these lines are not required in the algorithim to define if a point is in or out of domain.
                boundary_coordinates_1.append((X1,Y1))
                if(X2>X1):
                    for x in np.arange(X1,X2,space_laps):
                        horizontal.append((x,Y1))
                    

                elif(X1>X2):
                    boundary_coordinates_1.append((X2,Y2))
                    for x in np.arange(X1,X2,-(space_laps)):
                        horizontal.append((x,Y1))

            else:
                slope = (Y2 - Y1) / (X2 - X1)
                if (Y2 > Y1):
                    print("")
                    for y in np.arange(Y1 , Y2, space_laps ):
                        X = (((y - Y1)/slope) + X1)
                        boundary_coordinates_1.append((X, y))
                        #print("",(X,y))
            
                elif (Y1 > Y2):
                    print("")
                    for y in np.arange(Y1, Y2, -(space_laps)):
                        X = (((y - Y1)/slope) + X1)
                        boundary_coordinates_1.append((X, y))
                        #print("🕊️",(X,y))
                        
                                    
        else:
            pass

if (len(circle)!=0):
    for i in range(0,len(circle),1):
        cx=circle[i][0]
        cy=circle[i][1]
        r= circle[i][2]
        #print("@",cx,cy,r)
        if ((cx != "B" and cy !="R" and r != "E")):   
            #print("🎶",cx,cy,r)
            for i in np.arange((cy-r),(cy+r+space_laps),space_laps):
                io = i - cy
                rounded_io = round(io,8)
                theta=np.arcsin(rounded_io/r)
                r_the=round(theta,8)
                x=cx + r*np.cos(theta)
                y=cy + r*np.sin(theta)
                ###mirror####
                x2=cx-r*np.cos(theta)
                y2=cy+r*np.sin(theta)
                #print( "👍",rounded_io,"",theta,"",x,"",y)
                #print(x,"",y)
                #print(x2,"",y2)
                boundary_coordinates_1.append((x,y))
                boundary_coordinates_1.append((x2,y2))       

# Round the result
rounded_points = [(round(float(x), tol), round(float(y), tol)) for x, y in boundary_coordinates_1]
rounded_horizontal = [(round(float(x), tol), round(float(y), tol)) for x, y in horizontal]
rounded_verticals = [(round(float(x), tol), round(float(y), tol)) for x, y in vertical]
unique_rounded_points = list(set(rounded_points))

cad_ed = time.time()
cad_file_time = cad_ed - cad_st
print(Fore.LIGHTGREEN_EX + "Discretized boundary CAD file is ready !!! " + Style.RESET_ALL)
print(Fore.YELLOW + f"Discretized boundary CAD processing time = {cad_file_time}" + Style.RESET_ALL)


#-----vertex odd-even check-----#
print(Fore.LIGHTMAGENTA_EX + "Starting meshing process..." + Style.RESET_ALL)
mesh_st = time.time()
Even_vertex = []
if (len(line)!=0):
    for ip in range(0,len(brk_id),1):
        
        if (ip>0):
            end_point = np.sum(brk_id[:ip+1]) + ip
            start_point = np.sum([brk_id[:ip]]) + ip
        elif(ip==0):
            start_point = 0
            end_point = brk_id[ip]
        
        vertices = []
        # points before the start point
        vertices.append(line[end_point-3])
        vertices.append(line[end_point-2])
        for ihm in range(start_point,end_point,1):
            vertices.append(line[ihm])

        # point after the end point 
        vertices.append(vertices[3])
        print("😭",vertices)

        even_vertex = []
        for i in range(2,len(vertices)-2,1):
            if (abs(vertices[i][1] - vertices[i+1][1]) > 1e-5 and abs(vertices[i][0] - vertices[i+1][0]) > 1e-5):
                if (vertices[i-1][1] < vertices[i][1] and vertices[i+1][1] < vertices[i][1]):
                    even_vertex.append(vertices[i])
                    print(i,vertices[i][0],vertices[i][1])
                elif(vertices[i-1][1] > vertices[i][1] and vertices[i+1][1] > vertices[i][1]):
                    even_vertex.append(vertices[i])
                    print(i,vertices[i][0],vertices[i][1])
                else:
                    print("just passed")
            elif(abs(vertices[i][1] - vertices[i+1][1]) < 1e-5): # 
                if (abs(vertices[i][0] - vertices[i-1][0]) < 1e-5 and abs(vertices[i+1][0] - vertices[i+2][0]) < 1e-5 and ((vertices[i+2][1]  < vertices[i][1] < vertices[i-1][1]) or (vertices[i+2][1]  > vertices[i][1] > vertices[i-1][1]))):
                    even_vertex.append(vertices[i+1])
                    print("V:",vertices[i])
                else:
                    pass
            else:
                print("just passed 2",vertices[i])
                pass

        for im in range(0,len(even_vertex),1):
            unique_rounded_points.append(even_vertex[im]) 
            Even_vertex.append(even_vertex[im])
    
    print("even: ",Even_vertex)

print("----------------------------------------")
# print("u: ",unique_rounded_points)
# print("l: ",line)
if(len(circle)!=0):
    for i in range(0,len(circle),1):
        cx=circle[i][0]
        cy=circle[i][1]
        r= circle[i][2]
        if ((cx != "B" and cy !="R" and r != "E")):
            unique_rounded_points.append((cx,cy+r))
            unique_rounded_points.append((cx,cy-r))

s,d = zip(*unique_rounded_points)
plt.scatter(s,d , s=2)
plt.show()

#############################################################################################
Points = []
h = del_h 

for x in np.arange(0,x_max+del_h,h):
    for y in np.arange(0,y_max+del_h,h):
        Points.append((x,y))
print("🏃🏼‍♂️‍➡️: ",len(Points))
# time.sleep(5)
rounded_points_2 = [(round(float(x), tol), round(float(y), tol)) for x, y in Points] # giving same round of to points as given to boundary points 
#points = list(set(rounded_points_2) - (set(rounded_horizontal)|set(rounded_points))) # removing points that lies on boundary and thus are not required to be analyzed. 
boundary_set = set(rounded_horizontal) | set(rounded_points)

points = [p for p in rounded_points_2 if p not in boundary_set]

#------------------------------------ checking of odd-even intersections with respect to x------------------------------------------------#
interior_points_x=[]
for Y in np.arange(0,y_max+del_h,h):

    edge_points_x = []     # Stores data relating to boundary coordinates at particular Y value
    test_point_x = []      # Stores test points from complete space that have same particular Y value

    for i in range(0,len(unique_rounded_points),1):
        if (abs(unique_rounded_points[i][1] - Y) < tolerance):
            edge_points_x.append(unique_rounded_points[i])
        
                
    for j in range(0,len(points),1):
        if(abs(points[j][1] - Y) < tolerance):
            test_point_x.append(points[j])

    for k in range(0,len(test_point_x),1):
        counter_x=[]
        for w in range(0,len(edge_points_x),1):
            if(test_point_x[k][0] < edge_points_x[w][0]):
                counter_x.append("1")

        rem = len(counter_x)
        if (rem % 2 !=0):
            interior_points_x.append(test_point_x[k])
        else:
            pass

mesh_et = time.time()     
mesh_time = mesh_et - mesh_st  
print(Fore.LIGHTGREEN_EX + "Meshing Complete !!!" + Style.RESET_ALL)
print(Fore.YELLOW + f"Mesh processing time = {mesh_time}" + Style.RESET_ALL)

print("WAIT.....")
st = time.time()
unique_rounded_set = set(unique_rounded_points)
filtered_interior_x = [point for point in interior_points_x if point not in unique_rounded_set] # removing boundary points
filtered_interior_x_set = set(filtered_interior_x)  # creating hash-maps for O(1) step time complexity

filtered_exterior = list(set(points) - filtered_interior_x_set)  # All_the_points - interior_points = pureExteriorPoints + boundaryPoints

fill = []
for Y in np.arange(0,y_max+del_h,del_h):
    for u in range(0,len(unique_rounded_points),1):
        if (abs(unique_rounded_points[u][1] - Y) < tolerance):
            fill.append(unique_rounded_points[u])

filtered_exterior_x = list((set(filtered_exterior) | set(rounded_horizontal) |set(fill))) #  pureExteriorPoints + boundaryPoints + HorizontalPoints
et = time.time()
print("Done.")
print("Time taken = ",et-st)

#---------------------------------------------------------------ghost-node detection--------------------------------------------------------#
print(Fore.LIGHTMAGENTA_EX + "Starting Edge Detection..." + Style.RESET_ALL)
edge_st = time.time()

ghost_nodes = []
first_interface = []

A = torch.tensor(filtered_interior_x, dtype=torch.float64, device=device)   # shape (N, 2)
B = torch.tensor(filtered_exterior_x, dtype=torch.float64, device=device)   # shape (M, 2)
# print(filtered_interior_x[113],"<<")
for i in range(len(A)):
    A0 = A[i]                          # shape (2,)
    diff = B - A0                      # (M, 2)
    dist = torch.sqrt((diff[:,0])**2 + (diff[:,1])**2)   # GPU
    mask = (torch.abs(dist - del_h) < 1e-10)              # GPU
    # if (A0[0] == 7.0 and A0[1]==2.5):
    #     print(i,dist,mask)
    idx = torch.where(mask == True)[0]                   # GPU
    # if (A0[0] == 7.0 and A0[1]==2.5):
    #     print(idx)
    id = idx.cpu().numpy()
    if len(id) > 0:
        for j in range(0,len(id),1):
            ghost_nodes.append(filtered_exterior_x[id[j]])
            first_interface.append(filtered_interior_x[i])
    else:
        pass
print("?????",ghost_nodes)
yt = time.time()
print(">>>>>",yt-edge_st)
sorted_ghost_nodes = []
if (len(line)!=0):
    for i in range(0,len(line)-1,1):
        X1,Y1 = line[i][0],line[i][1]       # E
        X2,Y2 = line[i+1][0],line[i+1][1]   # F
        if ((X1 != "B" or Y1 !="R") and (X2 != "B" or Y2 !="R") ):
            print(X1,Y1,"and",X2,Y2)
            dist_EF = np.sqrt((X2-X1)**2 + (Y2-Y1)**2)      # length of EF
            if (abs(X2-X1) < tolerance):
                slope = 0
            else:
                slope = (Y2-Y1)/(X2-X1)
            alpha = np.arctan(slope)
            beta = np.pi - ((np.pi/2 + alpha))
            sub_sorted_ghost_node = []
            for j in range(0,len(ghost_nodes),1):
                X0,Y0 = ghost_nodes[j][0],ghost_nodes[j][1]
                # formula used = (|(y2 - y1)*x0 - (x2-x1)*y0 + (x2*y1 - y2*x1) |) / sqrt ((y2 - y1)**2 + (x2 - x1)**2)
                per_dist = abs(((Y2-Y1)*X0) - ((X2-X1)*Y0) + ((X2*Y1 - Y2*X1)))/(np.sqrt((Y2-Y1)**2 + (X2-X1)**2))
                sin_alpha = abs(per_dist/(np.sin(alpha)))
                sin_beta = abs(per_dist/(np.sin(beta)))
                rounded_sin_alpha = round(sin_alpha,tol)
                rounded_sin_beta = round(sin_beta,tol)
                dist_OE = np.sqrt((X0-X1)**2 + (Y0-Y1)**2)  # length OE
                dist_OF = np.sqrt((X0-X2)**2 + (Y0-Y2)**2)  # length OF
                # print("f: ",X0,Y0,"",per_dist,"",sin_alpha,"",sin_beta)
                # print("f: ",X0,Y0,"",per_dist,"",rounded_sin_alpha,"",rounded_sin_beta)
                # if (((rounded_sin_alpha < del_h) or (rounded_sin_beta < del_h)) and ((dist_OE <= dist_EF) and (dist_OF <= dist_EF))):
                #     sub_sorted_ghost_node.append((X0,Y0))
                if (((per_dist < np.sqrt(2*del_h**2))) and ((dist_OE <= dist_EF) and (dist_OF <= dist_EF))):
                    sub_sorted_ghost_node.append((X0,Y0))               
                else:
                    pass
            if(len(sub_sorted_ghost_node) > 0):
                sorted_ghost_nodes.append(sub_sorted_ghost_node)
            else:
                pass
        else:
            pass        
print("-------")
print("PPPP",sorted_ghost_nodes)
if(len(circle)!=0):
    for i in range(0,len(circle),1):
        cx,cy,r = circle[i][0],circle[i][1],circle[i][2]
        if ((cx != "B" and cy !="R" and r != "E")):
            sub_sorted_ghost_node = []
            for j in range(0,len(ghost_nodes),1):
                x0,y0 = ghost_nodes[j][0],ghost_nodes[j][1]
                dist_OE = np.sqrt((cx-x0)**2 + (cy-y0)**2)
                s = round(abs(r - dist_OE),tol)
                if( abs(cx - x0)< tolerance):
                    # slope = infinity
                    alpha = np.pi/2
                    beta = np.pi - (np.pi/2 + alpha)
                else:    
                    slope = (cy - y0)/(cx - x0)
                    alpha = np.arctan(slope)
                    beta = np.pi - (np.pi/2 + alpha)

                sin_alpha = np.sin(alpha)
                sin_beta = np.sin(beta)
                sx = abs(s/sin_alpha)
                sy = abs(s/sin_beta)
                rounded_sx = round(sx,tol)
                rounded_sy = round(sy,tol)
                        
                print("o: ",x0,y0,"",dist_OE,"",s,"",sx,sy)
                if ( (rounded_sx < del_h) or (rounded_sy < del_h)):
                    sub_sorted_ghost_node.append((x0,y0))
                else:
                    pass
            if(len(sub_sorted_ghost_node) > 0):
                sorted_ghost_nodes.append(sub_sorted_ghost_node)
            else:
                pass    
        else:
            pass
# print("MNJ",first_interface,len(first_interface))
# After analyzing and sorting out the ghost nodes...now we must get the corresponding first interfaces
sorted_first_interface = []
for i in range(0,len(sorted_ghost_nodes),1):
    sub_sorted_first_interface = []
    for j in range(0,len(sorted_ghost_nodes[i]),1):
        X0,Y0 = sorted_ghost_nodes[i][j][0],sorted_ghost_nodes[i][j][1]
        # print(X0,Y0)
        for k in range(0,len(first_interface),1):
            Xf,Yf = first_interface[k][0],first_interface[k][1]
            # if(Xf==7 and Yf==2.5):
            # print(k,Xf,Yf)
            f_dist_g = np.sqrt((X0 - Xf)**2 + (Y0 - Yf)**2)
            if (abs(f_dist_g - del_h ) < tolerance):
                sub_sorted_first_interface.append((Xf,Yf))
                break       # this "break" is important as it limits the finding to only one of the first interface w.r.t the ghost node

            else:
                pass
        # time.sleep(900)
    
    sorted_first_interface.append(sub_sorted_first_interface)

mirror = []
if (len(line) !=0 ):
    for i in range(0, len(sorted_ghost_nodes), 1):
        X1, Y1 = line[i][0], line[i][1]
        X2, Y2 = line[i+1][0], line[i+1][1]
        mirror_sub = []
        if ((X1 != "B" or Y1 !="R") and (X2 != "B" or Y2 !="R") ):
            # Handle different line cases
            if abs(X2 - X1) < 1e-10:  # Vertical line
                for j in range(len(sorted_ghost_nodes[i])):
                    Xg, Yg = sorted_ghost_nodes[i][j]
                    mirror_sub.append((2*X1 - Xg, Yg))
                    
            elif abs(Y2 - Y1) < 1e-10:  # Horizontal line
                for j in range(len(sorted_ghost_nodes[i])):
                    Xg, Yg = sorted_ghost_nodes[i][j]
                    mirror_sub.append((Xg, 2*Y1 - Yg))
                    
            else:  # Slanted line
                slope = (Y2 - Y1) / (X2 - X1)
                for j in range(len(sorted_ghost_nodes[i])):
                    Xg, Yg = sorted_ghost_nodes[i][j]
                    common = slope * (Xg - X1) - (Yg - Y1)
                    denominator = slope**2 + 1
                    Xm = Xg - (2 * slope * common) / denominator
                    Ym = Yg + (2 * common) / denominator
                    mirror_sub.append((Xm, Ym))
        
        mirror.append(mirror_sub)


# interpolation_points = []
# # detecting points for bilinear interpolation on mirror point
# for i in range(0,len(sorted_ghost_nodes),1):
#     X0,Y0 = line[i][0],line[i][1]
#     X1,Y1 = line[i+1][0],line[i+1][1]
#     print("#main")
#     if ((X1 != "B" or Y1 !="R") and (X2 != "B" or Y2 !="R") ):
#         if (abs(X1 - X0) < tolerance):
#             pass
#         elif(abs(Y1 - Y0) < tolerance):
#             slope = 0
#         else:
#             slope = (Y1-Y0)/(X1-X0)
#         print("sub_main")
#         sub_interpolation_points = []
#         for j in range(0,len(sorted_ghost_nodes[i]),1):
#             coords = []
#             Xg,Yg = sorted_ghost_nodes[i][j][0],sorted_ghost_nodes[i][j][1]
#             Xm,Ym = mirror[i][j][0],mirror[i][j][1]
#             if (abs(X1 - X0) < tolerance):  # vertical lines
#                 x = X0
#                 y = Yg
#             elif(abs(Y1 - Y0) < tolerance): # horizontal lines
#                 x = Xg
#                 y = Y0
#             else:                           # lines at angles != 90° and 180° 
#                 x = ((Yg - Y0)/slope) + X0
#                 y = Y0 + (slope*(Xg - X0))

#             xp = x - Xg
#             xp = round(xp,tol)
#             yp = y - Yg
#             yp = round(yp, tol)
#             print("#-map-#",xp,yp)
#             if (xp > 0):
#                 print("#1")
#                 alpha_x_1 = Xg + del_h
#                 alpha_x_2 = Xg + 2*del_h
#                 alpha_x_3 = Xg + 3*del_h
#                 alpha_x_1 = round(alpha_x_1,tol)
#                 alpha_x_2 = round(alpha_x_2,tol)
#                 alpha_x_3 = round(alpha_x_3,tol)
#                 # alpha_y_1 = Yg + del_h
#                 # alpha_y_2 = Yg + 2*del_h
#                 # alpha_y_3 = Yg + 3*del_h
#                 rt = time.time()
#                 if ((alpha_x_1, Yg) in filtered_interior_x_set): 
#                     coords.append((alpha_x_1,Yg))
#                 if ((alpha_x_2, Yg) in filtered_interior_x_set):
#                     coords.append((alpha_x_2,Yg))
#                 if((alpha_x_3, Yg) in filtered_interior_x_set):
#                     coords.append((alpha_x_3,Yg))
#                 et = time.time()
#                 print("<",et-rt,">")
                
#             if (xp < 0):
#                 print("#2")
#                 alpha_x_1 = Xg - del_h
#                 alpha_x_2 = Xg - 2*del_h
#                 alpha_x_3 = Xg - 3*del_h  
#                 # alpha_y_1 = Yg - del_h
#                 # alpha_y_2 = Yg - 2*del_h
#                 # alpha_y_3 = Yg - 3*del_h 
#                 alpha_x_1 = round(alpha_x_1,tol)
#                 alpha_x_2 = round(alpha_x_2,tol)
#                 alpha_x_3 = round(alpha_x_3,tol)
#                 rt = time.time()
#                 if ((alpha_x_1, Yg) in filtered_interior_x_set):    
#                     coords.append((alpha_x_1,Yg))
#                 if((alpha_x_2, Yg) in filtered_interior_x_set):
#                     coords.append((alpha_x_2,Yg))
#                 if((alpha_x_3, Yg) in filtered_interior_x_set):
#                     coords.append((alpha_x_3,Yg))
#                 et = time.time()
#                 print("<",et-rt,">")                     
#             if (yp > 0):
#                 print("#3")
#                 # alpha_x_1 = Xg - del_h
#                 # alpha_x_2 = Xg - 2*del_h
#                 # alpha_x_3 = Xg - 3*del_h  
#                 alpha_y_1 = Yg + del_h
#                 alpha_y_2 = Yg + 2*del_h
#                 alpha_y_3 = Yg + 3*del_h 
#                 alpha_y_1 = round(alpha_y_1,tol)
#                 alpha_y_2 = round(alpha_y_2,tol)
#                 alpha_y_3 = round(alpha_y_3,tol)
#                 rt = time.time() 
#                 if ((Xg, alpha_y_1) in filtered_interior_x_set):  
#                     coords.append((Xg,alpha_y_1))
#                 if ((Xg, alpha_y_2) in filtered_interior_x_set):
#                     coords.append((Xg,alpha_y_2))
#                 if((Xg, alpha_y_3) in filtered_interior_x_set):
#                     coords.append((Xg,alpha_y_3))
#                 et = time.time() 
#                 print("<",et-rt,">")
                
#             if (yp < 0):
#                 print("#4")
#                 # alpha_x_1 = Xg + del_h
#                 # alpha_x_2 = Xg + 2*del_h
#                 # alpha_x_3 = Xg + 3*del_h  
#                 alpha_y_1 = Yg - del_h
#                 alpha_y_2 = Yg - 2*del_h
#                 alpha_y_3 = Yg - 3*del_h 
#                 alpha_y_1 = round(alpha_y_1,tol)
#                 alpha_y_2 = round(alpha_y_2,tol)
#                 alpha_y_3 = round(alpha_y_3,tol)
#                 rt = time.time() 
#                 if ((Xg, alpha_y_1) in filtered_interior_x_set):  
#                     coords.append((Xg,alpha_y_1))
#                 if((Xg, alpha_y_2) in filtered_interior_x_set):
#                     coords.append((Xg,alpha_y_2))
#                 if((Xg, alpha_y_3) in filtered_interior_x_set):
#                     coords.append((Xg,alpha_y_3))
#                 et = time.time()
#                 print("<",et-rt,">")
#             if ((abs(Xg-Xm) < tolerance) and (abs(Yg-Ym) < tolerance)):
#                 coords.append((Xg,Yg))
#             else:
#                 pass
#             sub_interpolation_points.append(coords)
#         interpolation_points.append(sub_interpolation_points)   
#     else:
#         pass

# print(interpolation_points[0][7])

# Building second_interface
second_interface = set(filtered_interior_x) - set(first_interface)
# Convert set to list first, then to NumPy array
second_interface = np.array(list(second_interface), dtype=np.float64)
print("kok",second_interface)
# time.sleep(900)

# To use 3rd order FD formulations in solver
new_interface = []
A = torch.tensor(first_interface, dtype=torch.float64, device=device)   # shape (N, 2)
B = torch.tensor(second_interface, dtype=torch.float64, device=device)   # shape (M, 2)
for i in range(len(A)):
    A0 = A[i]                          # shape (2,)
    diff = B - A0                      # (M, 2)
    dist = torch.sqrt((diff[:,0])**2 + (diff[:,1])**2)   # GPU
    mask = (torch.abs(dist - del_h) < 1e-10)              # GPU
    idx = torch.where(mask == True)[0]                   # GPU
    id = idx.cpu().numpy()
    if len(id) > 0:
        for j in range(0,len(id),1):
            new_interface.append(second_interface[id[j]])
            
    else:
        pass

# Convert numpy arrays to sets of tuples
second_interface = {tuple(row.tolist()) for row in second_interface}
new_interface = {tuple(row.tolist()) for row in new_interface}

third_interface = set(second_interface) - set(new_interface)
second_interface = new_interface

third_interface = np.array(list(third_interface), dtype=np.float64)
second_interface = [tuple(s) for s in second_interface]
# To use 4th order FD formulations in solver
new_interface = []
A = torch.tensor(second_interface, dtype=torch.float64, device=device)   # shape (N, 2)
B = torch.tensor(third_interface, dtype=torch.float64, device=device)   # shape (M, 2)
for i in range(len(A)):
    A0 = A[i]                          # shape (2,)
    diff = B - A0                      # (M, 2)
    dist = torch.sqrt((diff[:,0])**2 + (diff[:,1])**2)   # GPU
    mask = (torch.abs(dist - del_h) < 1e-10)              # GPU
    idx = torch.where(mask == True)[0]                   # GPU
    id = idx.cpu().numpy()
    if len(id) > 0:
        for j in range(0,len(id),1):
            new_interface.append(third_interface[id[j]])
            
    else:
        pass

# Convert numpy arrays to sets of tuples
third_interface = {tuple(row.tolist()) for row in third_interface}
new_interface = {tuple(row.tolist()) for row in new_interface}

fourth_interface = set(third_interface) - set(new_interface)
third_interface = new_interface

third_interface = [tuple(s) for s in third_interface]
    
edge_et = time.time()
edge_time = edge_et - edge_st
print(Fore.LIGHTGREEN_EX + "Edge detection Complete !!!" + Style.RESET_ALL)
print(Fore.YELLOW + f"Total time to identify edges = {edge_time}" + Style.RESET_ALL)

print("")
print("")
#print(ghost_nodes)
print(Fore.BLUE + "Final mesh !!!" + Style.RESET_ALL)
print(f"total nodes: {len(filtered_interior_x)}")
print("")

print(sorted_ghost_nodes[0])
print(first_interface)

# print(filtered_interior_x)

# sorted_first_interface[0][0] = 20,2.1
# sorted_first_interface[0][200] = 20,2.1
# print(sorted_first_interface[0],len(sorted_first_interface[0]))


# Unzip coordinates for plotting
x, y = zip(*unique_rounded_points)
if (len(rounded_horizontal) !=0):
    g, h = zip(*rounded_horizontal)
c, d = zip(*points)
a, b = zip(*filtered_interior_x)
yama,lama = zip(*ghost_nodes)
vv = sorted_ghost_nodes[0]
eera,meera = zip(*vv)
vvc = sorted_first_interface[0]
dxc,vfc = zip(*vvc)
miros = mirror[0]
mxi,myi = zip(*miros)
xe,ye = zip(*filtered_exterior_x)

# xg = sorted_ghost_nodes[0][7][0]
# yg = sorted_ghost_nodes[0][7][1]
# xf = mirror[0][7][0]
# yf = mirror[0][7][1]
# ip_points = interpolation_points[0][7]
# xipn,yipn = zip(*ip_points)
xoh,yof = zip(*fourth_interface)
xol,yol = zip(*third_interface)
xok,yok = zip(*second_interface)
xoj,yoj = zip(*first_interface)



# Plotting
# plt.plot(x, y, 'r-', linewidth=2, label='Polygon Boundary')  # Red line
plt.scatter(x, y, color='red', s=1, label='Vertices')       # Red markers
if (len(rounded_horizontal) !=0):
    plt.scatter(g, h, color='red', s=1)
# plt.scatter(c, d, color ="blue", s= 5)
plt.scatter(a, b, color = "black", s = 5)       # x-marker
# plt.scatter(g, h, color = "black", s = 5)       # y-marker

plt.scatter(yama,lama,color = 'green', s = 10)
plt.scatter(eera,meera,color = 'blue', s = 10)
plt.scatter(dxc,vfc, color = "#E0E700", s = 5)
plt.scatter(xoh,yof, color = "#28B264", s=5)
plt.scatter(xol,yol, color = "#BE0095", s =5)
plt.scatter(xok,yok, color = "#D77F0C", s =5)
plt.scatter(xoj,yoj, color = "#0092C3", s =5)
# plt.scatter(mxi,myi,color = "#823A3A",s = 6)

# plt.scatter(xe,ye, s =7, color = 'blue')
# plt.scatter(xf,yf, s =7, color =  "#AF5858")
# plt.scatter(xipn,yipn, s =7, color = "#9200A5")


# Annotations and labels
# plt.title("Polygon Visualization with Red Markers", fontsize=14)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend()
plt.axis('equal')  # Fix aspect ratio
plt.show()


nx = int((x_max+del_h)/del_h) + 1
ny = int((y_max+del_h)/del_h) + 1

print("nx, ny = ",nx,ny)

path1 = r"D:/numerical computation/geometry meshing/Meshes/RAX_1"
path2 = r"D:/numerical computation/geometry meshing/Meshes/GAX_1"

# np.savez(path1, array1=filtered_interior_x)
# sorted_ghost_nodes = np.array(sorted_ghost_nodes, dtype=object)
# sorted_first_interface = np.array(sorted_first_interface, dtype=object)
# np.savez(path2, array1=sorted_ghost_nodes, array2 = sorted_first_interface)

# === Convert to numpy object arrays (important!) ===
sorted_ghost_nodes = np.array(sorted_ghost_nodes, dtype=object)
sorted_first_interface = np.array(sorted_first_interface, dtype=object)
# interpolation_points = np.array(interpolation_points, dtype=object)
mirror_array = np.array(mirror, dtype=object)
second_interface = np.array(second_interface, dtype=object)
third_interface = np.array(third_interface, dtype=object)
fourth_interface = np.array(fourth_interface, dtype=object)

# === Save paths ===
path1 = r"D:/numerical computation/geometry meshing/Meshes/RAX_1.npz"
path2 = r"D:/numerical computation/geometry meshing/Meshes/GAX_1.npz"

# # # === Save .npz files ===
np.savez(path1, array1=filtered_interior_x, del_h = del_h, nx = nx, ny = ny)
np.savez(
    path2,
    array1=sorted_ghost_nodes,
    array2=sorted_first_interface,
    # array3=interpolation_points,
    array4=mirror_array,
    array5=unique_rounded_points,
    array6=rounded_horizontal,
    array7=sorted_first_interface,
    array8=second_interface,
    array9=first_interface,
    array10=third_interface,
    array11=fourth_interface
)

print("✅ Geometry data saved successfully!")
# print(sorted_ghost_nodes)