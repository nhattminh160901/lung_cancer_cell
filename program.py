from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from datetime import datetime
import torch
import cv2
import os
import numpy as np

class FirstPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        Label(self,
              text="DETECT AND COUNTING CELLS",
              font=("Segoe UI Bold", 20)).pack(pady=10)
        Button(self, text="EXIT", command=lambda: controller.destroy()).place(x=1540, y=850)

        self.add_logo("logo\huet.png", 5, 590, (180, 180))
        self.add_logo("logo\chungcheng.png", 180, 590, (180, 180))
        self.add_logo("logo\pos.png", 1369, 350, (230, 230))
        
        self.original_image = None
        self.recheck = None
        
        
        self.model_name = "pt/v5version/v17/best.pt"
        self.model = torch.hub.load("yolov5", "custom", source="local", path=self.model_name, force_reload=True)

        self.function_buttons()
        self.image_frame()
        self.counting_frame()
        self.status_pos()

    def add_logo(self, path, x, y, size):
        img = Image.open(path)
        img = img.resize(size)
        
        logo = Label(self, text="")
        logo.image = ImageTk.PhotoImage(img)
        logo.configure(image=logo.image)
        logo.place(x=x, y=y)

    def function_buttons(self):
        frameOperationsImage = LabelFrame(self, text="File Manipulation Options", font=("Segoe UI Bold", 14))
        frameOperationsImage.place(x=10, y=50, height=170, width=350)

        buttonBrowseFile = Button(frameOperationsImage, text="Browse Image", width=12, command=self.open_image)
        buttonBrowseFile.place(x=5, y=0)

        self.labelFileName = Label(frameOperationsImage, text="File not choosen")
        self.labelFileName.place(x=110, y=3)

        buttonLoadmodel = Button(frameOperationsImage, text="Load Model", width=12, command=self.open_model)
        buttonLoadmodel.place(x=5, y=35)

        self.labelModelName = Label(frameOperationsImage, text=self.model_name)
        self.labelModelName.place(x=110, y=38)

        self.buttonCutImage = Button(frameOperationsImage, text="Cut Image", width=12, command=self.cut_image, state=DISABLED)
        self.buttonCutImage.place(x=5, y=70)

        self.buttonDaC = Button(frameOperationsImage, text="Detect and Count", width=15, command=self.detect_and_count, state=DISABLED)
        self.buttonDaC.place(x=110, y=70)

        self.buttonSavePos = Button(frameOperationsImage, text="Save to Pos", width=10, command=self.open_savepos_window, state=DISABLED)
        self.buttonSavePos.place(x=238, y=70)

        self.buttonUnlockScale = Button(frameOperationsImage, text="Unlock parameter Scale", width=18, command=self.unlock_scale, state=DISABLED)
        self.buttonUnlockScale.place(x=5, y=105)

        self.buttonOpenFolderPos = Button(frameOperationsImage, text="Open Pos Folder", width=14, command=self.openFolderPos)
        self.buttonOpenFolderPos.place(x=145, y=105)

        self.buttonResetFrameImage = Button(frameOperationsImage, text="Reset", width=8, command=self.reset)
        self.buttonResetFrameImage.place(x=259, y=105)

    def image_frame(self):
        self.frameImage = LabelFrame(self, text="Image Area", font=("Segoe UI Bold", 14))
        self.frameImage.place(x=370, y=50, height=815, width=1000)

        self.canvas = Canvas(self.frameImage)
        self.canvas.place(x=0, y=10)
        self.image_item = self.canvas.create_image(0, 0, anchor=NW)

        self.labelTotalc = Label(self.frameImage, text="", font=('Segoe UI', 16))
        self.labelTotaldc = Label(self.frameImage, text="", font=('Segoe UI', 16))

        labelHoughlinesThreshold = Label(self.frameImage, text="Hough Lines Threshold:")
        labelHoughlinesThreshold.place(x=20, y=748)
        self.houghlinesThreshold = Scale(self.frameImage, from_=100, to=350, orient="horizontal", resolution=1, command=self.update_image_scale)
        self.houghlinesThreshold.set(200)
        self.houghlinesThreshold.place(x=150, y=730)
        self.houghlinesThreshold.configure(state=DISABLED)

        labelDeviationAngel = Label(self.frameImage, text="Deviation Angel:")
        labelDeviationAngel.place(x=275, y=748)
        self.deviationAngel = Scale(self.frameImage, from_=-2, to=2, resolution=0.1, orient="horizontal", command=self.update_image_scale)
        self.deviationAngel.set(0)
        self.deviationAngel.place(x=367, y=730)
        self.deviationAngel.configure(state=DISABLED)

        labelCannyThreshold1 = Label(self.frameImage, text="Canny Threshold 1:")
        labelCannyThreshold1.place(x=490, y=748)
        self.cannyThreshold1 = Scale(self.frameImage, from_=0, to=50, orient="horizontal", resolution=1, command=self.update_image_scale)
        self.cannyThreshold1.set(30)
        self.cannyThreshold1.place(x=597, y=730)
        self.cannyThreshold1.configure(state=DISABLED)

        labelCannyThreshold2 = Label(self.frameImage, text="Canny Threshold 2:")
        labelCannyThreshold2.place(x=720, y=748)
        self.cannyThreshold2 = Scale(self.frameImage, from_=0, to=50, orient="horizontal", resolution=1, command=self.update_image_scale)
        self.cannyThreshold2.set(10)
        self.cannyThreshold2.place(x=825, y=730)
        self.cannyThreshold2.configure(state=DISABLED)

    def counting_frame(self):
        self.frameCounting = LabelFrame(self, text="Counting cells", font=("Segoe UI Bold", 14))
        self.frameCounting.place(x=1375, y= 50, height=300, width=220)

    def status_pos(self):
        self.statusDict = {"pos1":False,
                           "pos2":False,
                           "pos3":False,
                           "pos4":False,
                           "pos5":False,
                           "pos6":False,
                           "pos7":False,
                           "pos8":False}
        
        framePos = LabelFrame(self, text="Status of positions", font=("Segoe UI", 13))
        framePos.place(x=10, y=220, height=315, width=350)

        self.labelPos1 = Label(framePos, text="Pos 1:", font=("Segoe UI", 14))
        self.labelPos1.place(x=5, y=5)
        self.statusPos1 = Label(framePos, text="Waiting......", font=("Segoe UI", 14), fg="#FF0000")
        self.statusPos1.place(x=60, y=5)

        self.labelPos2 = Label(framePos, text="Pos 2:", font=("Segoe UI", 14))
        self.labelPos2.place(x=5, y=35)
        self.statusPos2 = Label(framePos, text="Waiting......", font=("Segoe UI", 14), fg="#FF0000")
        self.statusPos2.place(x=60, y=35)

        self.labelPos3 = Label(framePos, text="Pos 3:", font=("Segoe UI", 14))
        self.labelPos3.place(x=5, y=65)
        self.statusPos3 = Label(framePos, text="Waiting......", font=("Segoe UI", 14), fg="#FF0000")
        self.statusPos3.place(x=60, y=65)

        self.labelPos4 = Label(framePos, text="Pos 4:", font=("Segoe UI", 14))
        self.labelPos4.place(x=5, y=95)
        self.statusPos4 = Label(framePos, text="Waiting......", font=("Segoe UI", 14), fg="#FF0000")
        self.statusPos4.place(x=60, y=95)

        self.labelPos5 = Label(framePos, text="Pos 5:", font=("Segoe UI", 14))
        self.labelPos5.place(x=5, y=125)
        self.statusPos5 = Label(framePos, text="Waiting......", font=("Segoe UI", 14), fg="#FF0000")
        self.statusPos5.place(x=60, y=125)

        self.labelPos6 = Label(framePos, text="Pos 6:", font=("Segoe UI", 14))
        self.labelPos6.place(x=5, y=155)
        self.statusPos6 = Label(framePos, text="Waiting......", font=("Segoe UI", 14), fg="#FF0000")
        self.statusPos6.place(x=60, y=155)

        self.labelPos7 = Label(framePos, text="Pos 7:", font=("Segoe UI", 14))
        self.labelPos7.place(x=5, y=185)
        self.statusPos7 = Label(framePos, text="Waiting......", font=("Segoe UI", 14), fg="#FF0000")
        self.statusPos7.place(x=60, y=185)

        self.labelPos8 = Label(framePos, text="Pos 8:", font=("Segoe UI", 14))
        self.labelPos8.place(x=5, y=215)
        self.statusPos8 = Label(framePos, text="Waiting......", font=("Segoe UI", 14), fg="#FF0000")
        self.statusPos8.place(x=60, y=215)

        self.buttonResetStatus = Button(framePos, text="Reset Position", command=self.reset_status)
        self.buttonResetStatus.place(x=5, y=255)

        self.buttonOpenFolder = Button(framePos, text="Open Folder", command=self.openFolder)
        self.buttonOpenFolder.place(x=95, y=255)

        self.buttonCalculation = Button(framePos, text="Comprehensive Calculation", width=22, command= self.detect_and_comprehensive_calculation, state=NORMAL)
        self.buttonCalculation.place(x=176, y=255)

    def reset(self):
        self.frameImage.destroy()
        self.frameCounting.destroy()
        self.original_image = None
        self.recheck = None
        self.image_frame()
        self.counting_frame()

    def reset_status(self):
        self.statusDict = {"pos1":False,
                           "pos2":False,
                           "pos3":False,
                           "pos4":False,
                           "pos5":False,
                           "pos6":False,
                           "pos7":False,
                           "pos8":False}
        self.statusPos1.configure(text="Waiting......", fg="#FF0000")
        self.statusPos2.configure(text="Waiting......", fg="#FF0000")
        self.statusPos3.configure(text="Waiting......", fg="#FF0000")
        self.statusPos4.configure(text="Waiting......", fg="#FF0000")
        self.statusPos5.configure(text="Waiting......", fg="#FF0000")
        self.statusPos6.configure(text="Waiting......", fg="#FF0000")
        self.statusPos7.configure(text="Waiting......", fg="#FF0000")
        self.statusPos8.configure(text="Waiting......", fg="#FF0000")

    def openFolder(self):
        try:
            if not os.path.exists("runs"):
                os.mkdir("runs")
            path = os.getcwd()
            os.startfile(path+"/runs")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def openFolderPos(self):
        try:
            if not os.path.exists("img"):
                os.mkdir("img")
            path = os.getcwd()
            os.startfile(path+"/img")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        
    def open_savepos_window(self):
        windowSave = Toplevel(self)
        windowSave.title("Save into Position")
        windowSave.geometry('250x230')

        def dummy_select(event):
            # Allow the selection only if the selected item is not in the unselectable items list
            selected_index = listbox.curselection()
            if selected_index:
                item_index = int(selected_index[0])
                if item_index in listUnselectable:
                    listbox.selection_clear(0, END)  # Clear the selection

        def save_pos():
            selected_index = listbox.curselection()
            if selected_index:
                selected_item = listbox.get(ACTIVE)
                if selected_item==listkeys[0]:
                    self.statusDict["pos1"]=True
                    self.statusPos1.configure(text=self.labelFileName.cget("text"), fg="#008000")
                    img = self.areaImage.copy()
                    img = cv2.resize(img, (640,640), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(f"img/{listkeys[0]}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                elif selected_item==listkeys[1]:
                    self.statusDict["pos2"]=True
                    self.statusPos2.configure(text=self.labelFileName.cget("text"), fg="#008000")
                    img = self.areaImage.copy()
                    img = cv2.resize(img, (640,640), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(f"img/{listkeys[1]}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                elif selected_item==listkeys[2]:
                    self.statusDict["pos3"]=True
                    self.statusPos3.configure(text=self.labelFileName.cget("text"), fg="#008000")
                    img = self.areaImage.copy()
                    img = cv2.resize(img, (640,640), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(f"img/{listkeys[2]}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                elif selected_item==listkeys[3]:
                    self.statusDict["pos4"]=True
                    self.statusPos4.configure(text=self.labelFileName.cget("text"), fg="#008000")
                    img = self.areaImage.copy()
                    img = cv2.resize(img, (640,640), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(f"img/{listkeys[3]}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                elif selected_item==listkeys[4]:
                    self.statusDict["pos5"]=True
                    self.statusPos5.configure(text=self.labelFileName.cget("text"), fg="#008000")
                    img = self.areaImage.copy()
                    img = cv2.resize(img, (640,640), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(f"img/{listkeys[4]}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                elif selected_item==listkeys[5]:
                    self.statusDict["pos6"]=True
                    self.statusPos6.configure(text=self.labelFileName.cget("text"), fg="#008000")
                    img = self.areaImage.copy()
                    img = cv2.resize(img, (640,640), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(f"img/{listkeys[5]}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                elif selected_item==listkeys[6]:
                    self.statusDict["pos7"]=True
                    self.statusPos7.configure(text=self.labelFileName.cget("text"), fg="#008000")
                    img = self.areaImage.copy()
                    img = cv2.resize(img, (640,640), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(f"img/{listkeys[6]}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                elif selected_item==listkeys[7]:
                    self.statusDict["pos8"]=True
                    self.statusPos8.configure(text=self.labelFileName.cget("text"), fg="#008000")
                    img = self.areaImage.copy()
                    img = cv2.resize(img, (640,640), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(f"img/{listkeys[7]}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                windowSave.destroy()
            else:
                messagebox.showerror("Error", "Cant not save in this position")
                windowSave.destroy()
            
        labelSave = Label(windowSave, text="Choose Position to Save")
        labelSave.pack()

        listbox = Listbox(windowSave)
        listbox.pack()

        savepos = Button(windowSave, text="Save to Pos", command=save_pos)
        savepos.pack()

        listUnselectable = []

        listkeys = [key for key in self.statusDict]

        for i in range(len(listkeys)):
            listbox.insert(i, listkeys[i])
            if self.statusDict[listkeys[i]]:
                listbox.itemconfig(i, {"fg": "#008000"})
                listUnselectable.append(i)
            else:
                listbox.itemconfig(i, {"fg": "#FF0000"})
            
        listbox.bind('<<ListboxSelect>>', dummy_select)

    def open_model(self):      
        file_path = filedialog.askopenfilename(filetypes=[("Model", "*.pt")])
        rmfp = os.getcwd()
        rmfp = rmfp.replace(f"\\", "/")
        file_path = file_path.replace(rmfp, "").lstrip("/")
        if file_path:
            print(file_path)
            self.labelModelName.configure(text=file_path)
            self.model = torch.hub.load("yolov5", "custom", source="local", path=file_path)

    def open_image(self):
        filetypes = [("All Image Files", ("*.bmp", "*.ico", "*.jpeg", "*.jpg", "*.png", "*.ppm", "*.tif", "*.tiff", "*.gif", "*.xbm", "*.xpm")),
                    ("Windows Bitmap", "*.bmp"),
                    ("Icone ICO", "*.ico"),
                    ("Joint Photographic Experts Group", "*.jpeg"),
                    ("Joint Photographic Experts Group", "*.jpg"),
                    ("Portable Network Graphics", "*.png"),
                    ("Portable Pixmap", "*.ppm"),
                    ("Tagged Image File Format", "*.tif"),
                    ("Tagged Image File Format", "*.tiff"),
                    ("Graphic Interchange Format", "*.gif"),
                    ("X11 Pixmap", "*.xbm"),
                    ("X11 Pixmap", "*.xpm")]
        
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        if file_path and file_path!=self.recheck:
            self.labelFileName.configure(text=os.path.basename(file_path))

            self.labelTotalc.place_forget()
            self.labelTotaldc.place_forget()

            self.recheck=file_path
            self.original_image = cv2.imread(file_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.display_image(self.original_image.copy(), 70)
            
            
            self.houghlinesThreshold.configure(state=NORMAL)
            self.deviationAngel.configure(state=NORMAL)
            self.cannyThreshold1.configure(state=NORMAL)
            self.cannyThreshold2.configure(state=NORMAL)

            self.buttonCutImage.configure(state=DISABLED)
            self.buttonUnlockScale.configure(state=DISABLED)
            self.buttonDaC.configure(state=DISABLED)
            self.buttonSavePos.configure(state=DISABLED)

    def display_image(self, image, scale_percent):
        #scale_percent = 70 # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        self.canvas.configure(width=image.shape[1], height=image.shape[0])
        display_image = ImageTk.PhotoImage(image=Image.fromarray(image))
        self.canvas.itemconfig(self.image_item, image=display_image)
        self.canvas.image = display_image

    def update_image_scale(self, event):
        def filter_lines(lines):
            filtered_lines = []
            previous_r_line = lines[0]
            frist_line = True
            for r_theta in lines[1:]:
                if np.abs(r_theta[0][0]) - np.abs(previous_r_line[0][0])>150:
                    if (frist_line):
                        filtered_lines.append(previous_r_line)
                        frist_line = False
                    filtered_lines.append(r_theta)
                    print(1, r_theta)
                previous_r_line = r_theta
            return filtered_lines
        
        def sort_lines(lines):
            sorted_lines = sorted(lines, key=lambda line: line[0][0])
            filtered_lines = sorted_lines
            return filtered_lines

        def intersection(line1, line2):
            rho1, theta1 = line1[0]
            rho2, theta2 = line2[0]
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])
            x0, y0 = np.linalg.solve(A, b)
            x0, y0 = int(np.round(x0)), int(np.round(y0))
            return [x0, y0]
        
        def target_points(horizontal_lines, vertical_lines):
            points = []
            try:
                points.append(intersection(horizontal_lines[0],vertical_lines[0]))
                points.append(intersection(horizontal_lines[0],vertical_lines[4]))
                points.append(intersection(horizontal_lines[4],vertical_lines[4]))
                points.append(intersection(horizontal_lines[4],vertical_lines[0]))
                return points
            except:
                return points
            
        def filter_data(input_data, e=0):
            rad_e = e * np.pi/180
            target_value_1 = (np.pi+rad_e) / 2
            tolerance_1 = np.pi / 100
            lower_bound_1 = target_value_1 - tolerance_1
            upper_bound_1 = target_value_1 + tolerance_1

            target_value_2 = rad_e
            tolerance_2 = np.pi / 100
            lower_bound_2 = target_value_2 - tolerance_2
            upper_bound_2 = target_value_2 + tolerance_2

            target_value_3 = np.pi + rad_e
            tolerance_3 = np.pi / 100
            lower_bound_3 = target_value_3 - tolerance_3
            upper_bound_3 = target_value_3 + tolerance_3

            filtered_data = []
            horizontal_lines =[]
            vertical_lines = []

            for item in input_data:
                if lower_bound_1 <= item[0, 1] <= upper_bound_1:
                    #filtered_data.append(item)
                    horizontal_lines.append(item)
                else:
                    if lower_bound_2 <= item[0, 1] <= upper_bound_2:
                        #filtered_data.append(item)
                        vertical_lines.append(item)
                    else:
                        if lower_bound_3 <= item[0, 1] <= upper_bound_3:
                            #filtered_data.append(item)
                            vertical_lines.append(item)
            if (horizontal_lines):
                horizontal_lines = sort_lines(horizontal_lines)
                horizontal_lines = filter_lines(horizontal_lines)
            if (vertical_lines):
                vertical_lines = sort_lines(vertical_lines)
                vertical_lines = filter_lines(vertical_lines)
            filtered_data = horizontal_lines + vertical_lines

            return np.array(filtered_data), np.array(horizontal_lines), np.array(vertical_lines)
        
        if self.original_image is not None:
            img = self.original_image.copy()
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(img_gray, int(self.cannyThreshold1.get()), int(self.cannyThreshold2.get()))
            lines = cv2.HoughLines(edges, 1, np.pi/180, int(self.houghlinesThreshold.get()))
            filtered_lines_all = filter_data(lines, self.deviationAngel.get())
            filtered_lines = filtered_lines_all[0]
            #filtered_lines = lines

            for r_theta in filtered_lines:
                arr = np.array(r_theta[0], dtype=np.float64)
                r, theta = arr
                # Stores the value of cos(theta) in a
                a = np.cos(theta)
                # Stores the value of sin(theta) in b
                b = np.sin(theta)
                # x0 stores the value rcos(theta)
                x0 = a*r
                # y0 stores the value rsin(theta)
                y0 = b*r
                # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
                x1 = int(x0 + 1500*(-b))
                # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
                y1 = int(y0 + 1500*(a))
                # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
                x2 = int(x0 - 1500*(-b))
                # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
                y2 = int(y0 - 1500*(a))
                # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
                # (0,0,255) denotes the colour of the line to be
                # drawn. In this case, it is red.
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            self.points = target_points(filtered_lines_all[1], filtered_lines_all[2])
            if self.points:
                for point in self.points:
                    x, y = point
                    cv2.circle(img, (x, y), 5, (255, 0, 0), 5)
            cv2.circle(img, (0, 0), 5, (255, 0, 0), 5)

            if len(self.points)==4:
                self.houghlinesThreshold.configure(state=DISABLED)
                self.deviationAngel.configure(state=DISABLED)
                self.cannyThreshold1.configure(state=DISABLED)
                self.cannyThreshold2.configure(state=DISABLED)
                self.buttonCutImage.configure(state=NORMAL)
                self.buttonUnlockScale.configure(state=NORMAL)
            self.display_image(img, 70)

    def unlock_scale(self):
        self.houghlinesThreshold.configure(state=NORMAL)
        self.deviationAngel.configure(state=NORMAL)
        self.cannyThreshold1.configure(state=NORMAL)
        self.cannyThreshold2.configure(state=NORMAL)
        self.buttonUnlockScale.configure(state=DISABLED)

    def cut_image(self):
        self.buttonUnlockScale.configure(state=DISABLED)
        cut_img = self.original_image.copy()
        points = np.array([self.points], dtype=np.float32)
        # mask = np.zeros(cut_img.shape[0:2], dtype=np.uint8)
        
        # method 1 smooth region
        # cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
        # self.res = cv2.bitwise_and(cut_img, cut_img, mask=mask)

        width = int(max(np.linalg.norm(points[0][0] - points[0][1]), np.linalg.norm(points[0][2] - points[0][3])))
        height = int(max(np.linalg.norm(points[0][0] - points[0][3]), np.linalg.norm(points[0][1] - points[0][2])))
        new_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(points, new_points)

        self.areaImage = cv2.warpPerspective(cut_img, matrix, (width, height))
        self.display_image(self.areaImage, 90)
        self.buttonCutImage.configure(state=DISABLED)
        self.buttonDaC.configure(state=NORMAL)
        self.buttonSavePos.configure(state=NORMAL)

    def detect_and_count(self):
        self.buttonDaC.configure(state=DISABLED)
        self.main_image = self.areaImage.copy()
        self.main_image = cv2.resize(self.main_image, (640,640), interpolation=cv2.INTER_AREA)
        
        results = self.model(self.main_image)

        total_c = 0
        total_dc = 0
        
        def rectangle_puttext(image, label, xyxy, color):
            cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 1)
            cv2.putText(image, label, (int(xyxy[0]), int(xyxy[1])-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
          
        # for *pos, c, cls, in results.xyxy[0]:
        #     if c>0.05:
        #         if int(pos[0])==0 or int(pos[3])==640:
        #             ld+=1
        #         elif int(pos[1])==0 or int(pos[2])==640:
        #             ru+=1

        for *xyxy, conf, cls in results.xyxy[0]:
            if conf>0.55:
                if torch.round(xyxy[0])>=1 and torch.round(xyxy[1])>=1:
                    if self.model.names[int(cls)]=="cells":
                        print(xyxy)
                        total_c+=1
                        label = "cell"
                        rectangle_puttext(self.main_image, label, xyxy, (0,255,0))
                    elif self.model.names[int(cls)]=="die_cells":
                        print(xyxy)
                        total_dc+=1
                        label = "dead_cell"
                        rectangle_puttext(self.main_image, label, xyxy, (255,0,0))

        self.display_image(self.main_image, 111)
        # {conf:.2f}

        self.labelTotalc.configure(text=f"Total cell: {total_c}")
        self.labelTotalc.place(x=720, y=10)

        self.labelTotaldc.configure(text=f"Total dead cell: {total_dc}")
        self.labelTotaldc.place(x=720, y=40)

        messagebox.showinfo("Result of Detect", results)

    def detect_and_comprehensive_calculation(self):
        #if all(value for value in self.statusDict.values()):
        self.canvas.destroy()

        def rectangle_puttext(image, label, xyxy, color):
            cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 1)
            cv2.putText(image, label, (int(xyxy[0]), int(xyxy[1])-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        dictFileImg = {}

        x = 10
        y = 0        

        import glob
        for img in glob.glob("img/*.jpg"):
            cv_img = cv2.imread(img)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            dictFileImg[os.path.basename(img)] = cv_img

        list_total_c = []
        list_total_dc = []
        list_results = []

        for fileName in dictFileImg:
            img = dictFileImg[fileName]
            if x>800:
                x = 10
                y = 370

            total_c = 0
            total_dc = 0

            results = self.model(img)
            list_results.append(str(results))
            for *xyxy, conf, cls in results.xyxy[0]:
                if conf>0.55:
                    if torch.round(xyxy[0])>=1 and torch.round(xyxy[1])>=1:
                        if self.model.names[int(cls)]=="cells":
                            total_c+=1
                            label = "cell"
                            rectangle_puttext(img, label, xyxy, (0,255,0))
                        elif self.model.names[int(cls)]=="die_cells":
                            total_dc+=1
                            label = "dead_cell"
                            rectangle_puttext(img, label, xyxy, (255,0,0))
            
            dictFileImg[fileName] = img
            list_total_c.append(total_c)
            list_total_dc.append(total_dc)

            self.canvas = Canvas(self.frameImage)
            self.canvas.place(x=x, y=y)
            image_item = self.canvas.create_image(0, 0, anchor=NW)

            width = int(img.shape[1] * 37 / 100)
            height = int(img.shape[0] * 37 / 100)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

            self.canvas.configure(width=img.shape[1], height=img.shape[0])
            display_image = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.canvas.itemconfig(image_item, image=display_image)
            self.canvas.image = display_image

            labelFileName = Label(self.frameImage, text=f"File: {fileName}", font=("Segoe UI", 15))
            labelFileName.place(x=x+15, y=y+240)
            labelc = Label(self.frameImage, text=f"Total cell {str(total_c)}", font=("Segoe UI", 15))
            labelc.place(x=x+15, y=y+275)
            labeldc = Label(self.frameImage, text=f"Total dead cell {str(total_dc)}", font=("Segoe UI", 15))
            labeldc.place(x=x+15, y=y+310)
            x += 245
        
        messagebox.showinfo("Result of Detect", "\n".join(list_results))
        c = int(np.average(list_total_c)*1e4*2)
        dc = int(np.average(list_total_dc)*1e4*2)
        viability = c/(c+dc)*100

        labelcpml = Label(self.frameCounting, text=f"Live cell count:\n{c} cells/mL", font=("Segoe UI Bold", 15), justify="left")
        labelcpml.place(x=10, y=25)
        labeldcpml = Label(self.frameCounting, text=f"Dead cell count:\n{dc} cells/mL", font=("Segoe UI Bold", 15), justify="left")
        labeldcpml.place(x=10, y=105)
        estimateViability = Label(self.frameCounting, text=f"Estimate viability:\n{viability:.2f}%", font=("Segoe UI Bold", 15), justify="left")
        estimateViability.place(x=10, y=185)

        now = datetime.now()
        date_str = now.strftime("%d_%m_%Y")
        dt_str = now.strftime("%H_%M_%S")
        patch = "runs/"+date_str+"/"+ dt_str
        if not os.path.exists("runs"):
            os.mkdir("runs")
        if not os.path.exists("runs/"+date_str):
            os.mkdir("runs/"+date_str)
        if not os.path.exists(patch):
            os.mkdir(patch)

        dict_={"Total cell":c,
               "Total dead cell":dc,
               "Estimate viability":viability}
    
        for fileName in dictFileImg:
            cv2.imwrite(f"{patch}/{fileName}", cv2.cvtColor(dictFileImg[fileName], cv2.COLOR_RGB2BGR))

        import pandas as pd
        df = pd.DataFrame.from_dict([dict_])
        df.to_csv(f"{patch}/data.csv")
        # else:
        #     messagebox.showerror("Error", "Positions are not enough")

# class SecondPage(Frame):
#     def __init__(self, parent, controller):
#         Frame.__init__(self, parent)

class Application(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        
        window = Frame(self)
        window.pack()

        window.grid_columnconfigure(0, minsize=1600)
        window.grid_rowconfigure(0, minsize=900)

        self.frames = {}

        frame = FirstPage(window, self)
        self.frames[FirstPage] = frame
        frame.grid(row=0, column=0, sticky="nsew")

        # for F in (FirstPage, SecondPage):
        #     frame = F(window, self)
        #     self.frames[F] = frame
        #     frame.grid(row=0, column=0, sticky="nsew")
            
        self.show_frame(FirstPage)
        
    def show_frame(self, page):
        frame = self.frames[page]
        frame.tkraise()