# Adding new arguments to the pipeline

This example shows how to add new code to the libraries.

# Where to place your code

Within each addon module, you will find a comment block denoting where the user may begin to enter their own logic. An example of what to look for is shown below:

```
    ###################################
    #### User Provided Logic Below ####
    ###################################

    def channel_clean_logic(self,clean_method):
        """
        Update this function for the pipeline and direct handler to find new functions.

        Args:
            filetype (str): cleaning method to use
        """

        # Logic gates for different cleaning methods
        if clean_method.lower() == 'hup':
            self.HUP_clean()

    def HUP_clean(self):
        """
        Return the channel names according to HUP standards.
        Adapted from Akash Pattnaik code.
        Updated to handle labels not typically generated at HUP (All leters, no numbers.)
        """

        self.clean_channel_map = []
        for ichannel in self.channels:
            regex_match = re.match(r"(\D+)(\d+)", ichannel)
            if regex_match != None:
                lead        = regex_match.group(1).replace("EEG", "").strip()
                contact     = int(regex_match.group(2))
                new_name    = f"{lead}{contact:02d}"
            else:
                new_name = ichannel.replace("EEG","").replace("-REF","").strip()
            self.clean_channel_map.append(new_name.upper())
```
