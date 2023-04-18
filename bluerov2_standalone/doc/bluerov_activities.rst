.. _activities_bluerov_sim:

BlueROV2 Activity Definition
-----------------------------

.. _fig_bluerov2_architecture:

.. figure:: ./imgs/webgme_model.png
    :width: 500px
    :align: center
    :figclass: align-center

    System architecture of BlueROV2 UUV


.. mdinclude:: bluerov_activities.md


Evaluation & metrics
-----------------------

The `bluerov_eval.py` node does some basic realtime (`1Hz` update) evaluation, saved as `results/bluerov_evaluation.txt`. *(note: pipeline related nodes are still running in other, eg. waypoint missions also)*

.. code-block:: none

    [bluerov_eval.py]

    Evaluation:
    =========
    * Simulation time total:    : 151 [sec]
    ------ [ Pipe tracking metrics] ------
    * Pipeline detection ratio  : 1
    * Average pipeline distace  : 22.2451621597 [meter]
    * Tracking error ratio      : 0.0219278140944
    * Semseg bad, not used      : 3 [sec]
    * Pipeline not in view      : 2 [sec]
    * LEC2 AM not triggered     : 0 [sec]
    -------- [ Waypoint metrics] --------
    * Average cross track error : 0 [m]
    * Time to complete          : -1 [sec]
    -------- [ Degradation GT info ] --------
    * Degradation starting time : -1 [sec]
    * Degraded thruster id      : -1
    * Degraded efficiency       : -1
    -------- [ FDI LEC info ] --------
    * Thruster Reallocation     : 53.67 [sec]
    * FDI Degraded thruster id  : 1.0
    * FDI Degraded efficiency   : 75.0


