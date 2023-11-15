# Data Visualizations

## EDF Viewer
```
utils/visualization/edf_viewer.py --file ../../user_data/BIDS/BIDS/sub-0014/ses-preimplant04/eeg/sub-0014_ses-preimplant04_task-task_run-06_eeg.edf --sleep_wake_power ../../user_data/derivative/sleep_state/timeseries_association/reference.pickle
```

### Example Views
The default view for an EDF file following channel name cleanup and montaging might look like the following:
![default_setting](images/edf_viewer_00.png)

In this next example, by using the `t` button, we can highlight timepoints that might be of interest. In this case, we have data of interest in a sleep/wake study.
![first_target](images/edf_viewer_01.png)

If we press t again, we can go to another series of timepoints of interest. We can also have multiple categories of interesting points, which results in different highlighting colors.
![second_target](images/edf_viewer_02.png)

If we wish to zoom in on the data, we can left click to define an area of interest:
![zoom_window](images/edf_viewer_03.png)

And by pressing `z` we can zoom in on that region (and reset the zoom by pressing `r`):
![zoom](images/edf_viewer_04.png)

Finally, if we wish to look at just one channel closely, we can hover over the timeseries and press `e` to get a full plot zoom in:
![single_view](images/edf_viewer_05.png)
