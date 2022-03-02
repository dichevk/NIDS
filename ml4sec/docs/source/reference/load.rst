Load data
=========
We load data from any ``infile.csv`` to a pandas DataFrame with the following method. All packets will be sorted by the column ``timestamp``.

.. automethod:: ml4sec.Assignment.load

For this assignment the data loaded from `benign.csv` and `unknown.csv` contain summaries of network packets from various TCP/UDP flows.
We use the method ``Assignment.flows()`` to extract these flows.
This method returns a list of ``(flow-tuple, pd.DataFrame)`` and a nunmpy array of labels for each flow.
Where the ``flow-tuple`` consists of ``(protocol, src, sport, dst, dport)``, and the dataframe contains all packets within the flow.

.. automethod:: ml4sec.Assignment.flows
