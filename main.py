from program import *

app = Application()
app.geometry("1600x900")
app.title("Electrochemical Measurement Application")
app.iconbitmap("cell.ico")
app.resizable(False,False)
app.mainloop()

# ico = Image.open("images/huet.png")
# photo = ImageTk.PhotoImage(ico)
# app.wm_iconphoto(False, photo)