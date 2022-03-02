import json
import numpy as np
import requests

########################################################################
#                                Submit                                #
########################################################################

def submit(self, student_number, highscore_name=None):
    """Submit your prediction to the server to check if the prediction of your
        unknown data is correct. The result will print whether you passed the
        assignment or not.

        Parameters
        ----------
        student_number : string
            Your student number as a string, e.g., 's1234567'
            Please use your actual student number, we have build in checks
            to prevent froud.

        highscore_name : string, optional
            If given, your performance will be published on the highscore
            page under this name.
        """
    # Use benign data for training and unknown data for testing
    # Note that we use the minimum and maximum values from the training data to
    # scale the test data
    X_train, minimum, maximum = self.scale(self.feature_matrix(self.flows_benign))
    X_test , minimum, maximum = self.scale(
        self.feature_matrix(self.flows_unknown),
        minimum = minimum,
        maximum = maximum,
    )

    # Fit training data and predict testing data
    self.NIDS.fit(X_train)
    prediction = self.NIDS.predict(X_test)

    # Cast prediction to list of integers
    prediction = np.asarray(prediction, dtype=int).tolist()

    # Create JSON data to send
    data = {
        "prediction"    : prediction,
        "student_number": student_number,
        "name"          : highscore_name
    }

    # Submit data
    r = requests.post("https://vm-thijs.ewi.utwente.nl/ml/submit/", json=data)
    # Read response
    try:
        r = json.loads(r.text)
    except:
        print("SERVER SIDE ERROR: please contact t.s.vanede@utwente.nl.")
        r = {}

    # Print errors if any
    if len(r.get('errors', [])) > 0:
        print("Errors found:")
        for error in r.get('errors'):
            print("    - {}".format(error))
        print()

    # Show performance
    self.show_report(
        tpr      = r.get('tpr'     , float('NaN')),
        tnr      = r.get('tnr'     , float('NaN')),
        fpr      = r.get('fpr'     , float('NaN')),
        fnr      = r.get('fnr'     , float('NaN')),
        accuracy = r.get('accuracy', float('NaN')),

        precision = r.get('precision', float('NaN')),
        recall    = r.get('recall'   , float('NaN')),
        f1_score  = r.get('f1-score' , float('NaN')),
    )

    # Check if you passed
    if r.get('accuracy', 0) > 0.90 and r.get('fpr', 1) < 0.05:
        print("You have passed the assignment!")
    else:
        print("You have not yet passed the assignment.")
