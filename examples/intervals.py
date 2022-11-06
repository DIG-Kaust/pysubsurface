r"""
Intervals
=========

This example shows how a :class:`pysubsurface.objects.Intervals`
can be defined and visualized. Such an object is particularly useful for
defining stratigraphic columns with hierarchies between groups, formations,
and subformations. In this case we will create a silly hiererical structure of
the World.
"""
import matplotlib.pyplot as plt
from pysubsurface.objects import Intervals

plt.close('all')

###############################################################################
# We first initialize the object, after we add our intervals manually
interval = Intervals()
interval.add_interval('World', 'Ottawa', 'Sidney', 0, 'black', order=0)
interval.add_interval('America', 'Ottawa', 'Bejing', 1, 'red', order=0, parent='World')
interval.add_interval('Asia', 'Bejing', 'KL', 1, 'green', order=1, parent='World')
interval.add_interval('Oceania', 'KL', 'Sidney', 1, 'blue', order=2, parent='World')
interval.add_interval('Canada', 'Ottawa', 'NY', 2, 'red', order=0, parent='America')
interval.add_interval('USA', 'NY', 'Bejing', 2, 'green', order=1, parent='America')
interval.add_interval('Cina', 'Bejing', 'HK', 2, 'blue', order=0, parent='Asia')
interval.add_interval('Malaysia', 'HK', 'KL', 2, 'blue', order=1, parent='Asia')
interval.add_interval('NewZealand', 'KL', 'Auckland', 2, 'yellow', order=0, parent='Oceania')
interval.add_interval('Australia', 'Auckland', 'Sidney', 2, 'cyan', order=1, parent='Oceania')

###############################################################################
# Finally, we visualize our stratigraphic column
interval.view()