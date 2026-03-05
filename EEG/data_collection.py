from psychopy import visual, core, event
import serial

# --- Setup Serial ---
PORT = 'COM8'

mmbts = serial.Serial()
mmbts.port = PORT
mmbts.open()

# --- Window ---
win = visual.Window(size=[1200, 800], units='pix', color='black')
message = visual.TextStim(win, text='', color='white', height=40)

clock = core.Clock()
display_time = 1.0   # show message for 1 second
show_message = False

running = True
trigger_to_send = 1
while running:
    
    keys = event.getKeys()
    
    # ESC → quit immediately
    if 'escape' in keys:
        running = False
    
    # SPACE → send trigger
    if 'space' in keys:
        trigger_to_send = trigger_to_send + 1   # choose your trigger number
        win.callOnFlip(mmbts.write, bytes([trigger_to_send]))
        win.flip()  # flip required for callOnFlip to execute
        show_message = True
    if show_message:
        message.text = "Sent trigger"
        message.draw()
        if clock.getTime() > display_time:
            show_message = False
    win.flip()

# --- Clean Exit ---
mmbts.close()
win.close()
core.quit()