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

wait_clock = core.Clock()
msg_clock = core.Clock()

display_time = 2.0        # show "Sent trigger" for 2 seconds
wait_duration = 30.0       # wait 30 seconds after space press

waiting = False            # True while counting down 30s
show_message = False       # True while showing "Sent trigger"
running = True
trigger_to_send = 1

while running:
    keys = event.getKeys()

    # ESC → quit immediately
    if 'escape' in keys:
        running = False

    # SPACE → start 30s wait (only if not already waiting or showing message)
    if 'space' in keys and not waiting and not show_message:
        win.color = 'black'
        waiting = True
        wait_clock.reset()
        win.callOnFlip(mmbts.write, bytes([trigger_to_send]))
        win.flip()
        trigger_to_send += 1

    # After 30s → send trigger
    if waiting and wait_clock.getTime() >= wait_duration:
        win.color = 'red'
        win.callOnFlip(mmbts.write, bytes([trigger_to_send]))
        win.flip()
        trigger_to_send += 1
        waiting = False
        show_message = True
        msg_clock.reset()


    # Show "Sent trigger" for 2 seconds, then return to idle
    if show_message:
        message.text = "Sent trigger"
        message.draw()
        if msg_clock.getTime() > display_time:
            show_message = False

    win.flip()

# --- Clean Exit ---
mmbts.close()
win.close()
core.quit()