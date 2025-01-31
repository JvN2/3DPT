# 3DPT

This code accompanies the paper: "Multiplexed Nanometric 3D Tracking of Microbeads using a FFT-Phasor Algorithm", by Brouwer, Hermans and van Noort (https://www.biorxiv.org/content/10.1101/763706v1).

This software was written in LabVIEW 2018.

Use 'Main TrackingSimulation.vi' to start Lorentz Mie Scattering Theory simulations that emulate Magnetic Tweezers data.

Typical intended use:
1) Fill out 'LUT definition' and 'Ref definition' parameters that define the tracking for radial profile and Phasor tracking
2) Fill out 'LM parameters' that define the holographic images
3) Run from File menu: 'Setup>Calibration' to setup reference images and LUT
5) Fill out 'Simulation definition' to define a tracking experiment
6) Run from File menu: 'Setup>Simulation' to simulate measurement data
7) Run from File menu: 'Action>Run' to start tracking using both methods

For analyis visually inspect timetraces or use slider 'Display frame' to go through individual ROIs.

Optionally:
8) Run from File menu: 'File>Save>Traces' to store time traces and the currently visible ROI
9) Run from File menu: 'File>Save>Results' to store accuracy statistics. You can mannually fill out 'Variable' and 'Value' to include a parameter in the data file. Rows will be appended if an existing file is selected


Stand alone use of code:
For using the 3DPT in your own tracking application: Open the block diagram and locate the 'Run' case in the 'Menu Selection (User)' event. The left For loop implements the tracking.

Input reference images are generated by the 'CreateRefImages2.vi'

Calibration of the polynomals requires experimental data. An example of how these are processed can be found in 'CreateLUTRef.vi'. The top part of the code calibrates the 3DPT reference images. Full calibration setup can be found in 'Main TrackingSimulation.vi' in the block diagram: locate the 'Calibration' case in the 'Menu Selection (User)' event.
