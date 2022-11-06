.. _installation:

Installation
============

**PySubsurface** is developed in *Python3* and requires **Python 3.8 or greater**.


Step-by-step installation
-------------------------

Clone the repository:

.. code-block:: bash

    >> git clone git@github.com:DIG-KAUST/pysubsurface.git

Create an environment using the following command:

.. code-block:: bash

   >> make install

or in developer mode

.. code-block:: bash

   >> make install_dev

To ensure that everything has been setup correctly, run tests:

.. code-block:: bash

    >> make tests

Make sure no tests fail, this guarantees that the installation has been successfull.

Try to also build the documentation:

.. code-block:: bash

    >> make doc

If everything works fine, you are good to go! Add code, documentation or examples and follow the guidelines for
best practice code standards within *PySubsurface* in :ref:`contributing` page.