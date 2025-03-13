import numpy as np

# corners = np.array([[[389., 81.],[407., 81.],[407., 97.],[389., 98.]], [[300., 80.],[400., 80.],[400., 100.],[390., 100.]]])

def gate_position_cf(corners):
    # # define the video dimensions (720p) and centre
    # video_dim = np.array([720, 960])
    # video_centre = video_dim / 2

    # find the mean position of the markers
    corners1 = np.array(corners[0][0])
    corners2 = np.array(corners[1][0])

    marker1 = np.array( [ int(np.mean(corners1[:,0])), int(np.mean(corners1[:,1])) ] )
    marker2 = np.array( [ int(np.mean(corners2[:,0])), int(np.mean(corners2[:,1])) ] )
    # gate_markers = np.array([marker1, marker2])

    # centre of the gate, with markers at two opposite corners
    # gate_centre = np.mean(gate_markers, axis=0)
    gate_centre = np.array([(marker1[0]+marker2[0])//2,(marker1[1]+marker2[1])//2])


    # round to nearest pixel
    gate_centre = np.array([int(gate_centre[0]), int(gate_centre[1]) + 100])

    print(gate_centre)



    return gate_centre, marker1, marker2

    # # translation from the video centre to gate centre (rounded to whole pixels)
    # print(f'video centre = {video_centre}, gate centre = {gate_centre}')
    # translate = gate_centre - video_centre

    # translate = np.array([int(translate[0]),int(translate[1])])

    # return translate

# print(gate_position_cf(corners))
