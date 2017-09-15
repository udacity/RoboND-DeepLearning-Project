#### current debugging methods

The general idea is to varify that the data being sent to the quad matches what state the of simulator
Likewise we varify that the data being received from the quad is the same as the quad state

run `python server.py` and then `python tests_asof_sept3.py`and instructions to create a marker box using the center pixel coordinates of the image will be intermittently sent to the simulator.

The way I am debugging is to run the simulation in unity(I build for everything else). In the unity inspector panel,
left click on the `i` symbol located in the upper left corner, and select `debug` this will give access to additional information about the world state


Select the followme option on the opening menu of the sim. after the sim has loaded I generally navigate so the quad
is directly facing a building. then use the mouse wheel to change the viewpoint to be decently far from the quad. A green block is supposed to appear directly in front of the quad, or in the center of image on the lower right(the block doesn't show up in this image though)

You can check the quads coordinates by selecting `Quad Drone` in the hierarchy, and the markers will appear in the hierchy named after there ids(a five digit integer). Selecting these will allow for access to the pose of the marker. 

if the marker, and the drone are in the place python code thinks its supposed to be, then likely the python code is creating the marker position incorrectly, otherwise is the there is disagreement in any of the positions between the inspector in unity and the python code, then this either has to be found and fixed or temporarily compensated for to allow further debugging. 

