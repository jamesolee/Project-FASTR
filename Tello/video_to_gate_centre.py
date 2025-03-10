import numpy as np

corners = np.array([[[389., 81.],[407., 81.],[407., 97.],[389., 98.]], [[300., 80.],[400., 80.],[400., 100.],[390., 100.]]])

def gate_position_cf(corners):
    # define the video dimensions (720p) and centre
    video_dim = np.array([1280, 720])
    video_centre = video_dim / 2

    # find the mean position of the markers
    marker1 = np.array( [ np.mean(corners[0][:,0]), np.mean(corners[0][:,1]) ] )
    marker2 = np.array( [ np.mean(corners[1][:,0]), np.mean(corners[1][:,1]) ] )
    gate_markers = np.array([marker1, marker2])

    # centre of the gate, with markers at two opposite corners
    gate_centre = np.mean(gate_markers, axis=0)
    print(gate_centre)

    # translation from the video centre to gate centre
    translate = gate_centre - video_centre
    return translate

print(gate_position_cf(corners))
