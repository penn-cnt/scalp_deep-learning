# Directly accessing the core libraries

There are two methods for using the core libraries kept within this framework. We recommend the first method, but a brief description of the second is provided for edification.

## Direct Inputs Method

Each addon module includes a direct_inputs method to pass each module data and obtain its relevant output. An example for the montage class is as follows:
```
    def direct_inputs(self,DF,montage):
        """
        Method for getting channel montages directly outside of the pipeline environment.

        Args:
            DF (datafram): Dataframe to get montage for.
            montage (str): Montage to perform

        Returns:
            dataframe: New dataframe with montage data and channel names
        """
        
        # Save the user provided dataframe
        self.dataframe_to_montage = DF

        # Apply montage logic
        montage_data = self.channel_montage_logic(montage)

        return PD.DataFrame(montage_data,columns=self.montage_channels)
```
