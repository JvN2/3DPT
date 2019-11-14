# 3DPT

Use Main TrackingSimulation.vi to start Lorentz Mie Scattering Theory simulations that emulate Magnetic Tweezers data.

Typical intended use:
1) Fill out LUT definition and Ref definition parameters that define the tracking for radial profile and Phasor tracking
2) Fill out LM parameters that define the holographic images
3) Run from File menu: Setup>Calibration to setup reference images and LUT
5) Fill out Simulation definition to define a tracking experiment
6) Run from File menu: Setup>Simulation to simulate measurement data
7) Run from File menu: Action>Run to sstart tracking using both methods

For analyis visually inspect timetraces or use slide 'Display frame' to go through individual ROIs

Optionally:
8) Run from File menu: File>Save>Traces to store time traces and the currenly visible ROI
9) Run from File menu: File>Save>Results to store accuracy statistics. You can mannually fill out 'Variable' and 'Value' to include a parameter in the data file. Rows will be appended if an existing file is selected


