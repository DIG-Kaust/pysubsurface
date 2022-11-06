import logging

import os
import re
import shutil
import copy
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from distutils.dir_util import copy_tree
from deprecated import deprecated
from pysubsurface.utils.utils import change_name_for_unix
from pysubsurface.utils.stats import average_stats
from pysubsurface.objects.intervals import Intervals
from pysubsurface.objects.picks import Picks
from pysubsurface.objects.well import Well
from pysubsurface.objects.polygon import Polygon
from pysubsurface.objects.polygonset import PolygonSet
from pysubsurface.objects.interpretation import Interpretation, Ensemble
from pysubsurface.objects.seismic import Seismic
from pysubsurface.objects.seismicirregular import SeismicIrregular
from pysubsurface.objects.prestackseismic import PrestackSeismic
from pysubsurface.objects.fault import Fault
from pysubsurface.objects.facies import Facies

from pysubsurface.visual.utils import _discrete_cmap, \
    _pie_in_axes, _seismic_polarity
from pysubsurface.visual.cmap import *


try:
    from IPython.display import display
    ipython_flag = True
except:
    ipython_flag=False


def _create_figpath(rootdir, subdir):
    """Create subdir if not available
    """
    figpath = os.path.join(rootdir, subdir)
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    return figpath


class Project:
    """Project object.

    A Project object is a container for various datatypes that can be used
    together for various visualization and analysis. It generally contains
    a :class:`pysubsurface.objects.Interval` object, a :class:`pysubsurface.objects.Picks`
    object, a list of a :class:`pysubsurface.objects.Well` objects, a list of
    :class:`pysubsurface.objects.Seismic` and/or a
    :class:`pysubsurface.objects.SeismicIrregular` objects, a lists of
    :class:`pysubsurface.objects.Interpretation` objects (containing various sets of
    :class:`pysubsurface.objects.Surface` objects
    representing a unique interpretation), and a list of
    :class:`pysubsurface.objects.Surface` objects containing generic surfaces
    (e.g., amplitude maps, thickness maps) not directly associated to a
    specific interpretation.

    At initialization the :class:`pysubsurface.objects.Project` object
    will contains only an empty :class:`pysubsurface.objects.Interval' object
    and the :class:`pysubsurface.objects.Picks` object. The
    :class:`pysubsurface.objects.Project` class provides methods to add data with the
    aim of creating clear, reproducibles workflow when setting
    up a pysubsurface project.

    .. note:: Input data must be organized in a rigid and standardized data
      tree for the :class:`pysubsurface.objects.Project` class to be able to load those
      data internally. Follow the guidelines in :ref:`datastructure` when
      setting up a data structure for pysubsurface.

    Parameters
    ----------
    kind : :obj:`str`, optional
        Project kind, ``local`` when data are stored locally in a folder. Other options are not available
    projectsetup : :obj:`str`, optional
        Name of yaml file containing setup for project.
        Alternatively, for local projects only, ``projectdir`` and
        ``projectname`` can be provided directly
    projectdir : :obj:`str`, optional
        Project data directory (where input files are located following
        the folder structure convention of pysubsurface)
    projectname : :obj:`str`, optional
        Name of project
    figdir : :obj:`str`, optional
        Directory where figures are stored locally
    docfigdir : :obj:`str`, optional
        Directory in PTCdoc folder where local figures are re-mapped by running
        :func:`pysubsurface.objects.Project.mapdocfigs`
    warnings : :obj:`bool`, optional
        Display warnings
    verb : :obj:`bool`, optional
        Verbosity

    """
    def __init__(self, kind='local', projectsetup=None,
                 projectdir=None, projectname=None, figdir='.', docfigdir=None,
                 warnings=False, verb=False):
        if warnings:
            logging.getLogger().setLevel(logging.WARNING)
        else:
            logging.getLogger().setLevel(logging.ERROR)

        self._kind = kind
        if kind == 'local' and (projectdir is None or projectname is None):
            raise ValueError('provide projectdir and project name '
                             'for kind=local')
        if kind == 'local':
            self.projectdir = projectdir
            self.projectname = projectname
            self.figdir = figdir
            self.docfigdir = docfigdir
        else:
            raise NotImplementedError('kind must be local...')

        self._projectname_nospace = self.projectname.replace(" ", "")

        # create local figure directory
        if not os.path.exists(self.figdir):
            os.makedirs(self.figdir)
        if self.docfigdir is not None:
            if not os.path.exists(self.docfigdir):
                os.makedirs(self.docfigdir)

        # create interval object
        self.intervals = Intervals(kind=self._kind)

        # find all pick files
        if kind == 'local':
            pickfiles = \
                os.listdir(os.path.join(self.projectdir, 'Well', 'Picks'))
            # avoid including temp files
            pickfiles = [pickfile for pickfile in pickfiles if pickfile[-1] != '~']
            # read picks
            self.picks = Picks(os.path.join(self.projectdir,
                                            'Well', 'Picks', pickfiles[0]),
                               kind=self._kind, verb=verb)
            if len(pickfiles) > 1:
                for pickfile in pickfiles[1:]:
                    otherpicks = Picks(os.path.join(self.projectdir,
                                                    'Well', 'Picks', pickfile),
                                       kind=self._kind, verb=verb)
                    self.picks.concat_picks(otherpicks)
        else:
            self.picks = Picks('', kind=self._kind, verb=verb)

        # initialize horizon, map and seismic containers
        self.wells = {}
        self.polygons = {}
        self.polygonsets = {}
        self.horizonsets = {}
        self.maps = {}
        self.faults = {}
        self.seismics = {}
        self.preseismics = {}
        self.aveprops = {}
        self.facies = {}
        self._verb = verb

    def _collect_intervals(self):
        """Collect intervals for all wells in unique dataframe
        """
        # collect intervals
        def add_wellname_to_invervals(well, wellname):
            intervals = well.intervals
            if intervals is not None:
                intervals['wellname'] = wellname
            return intervals

        intervals = [add_wellname_to_invervals(self.wells[wellname], wellname)
                     for wellname in self.wells.keys()]
        intervals = pd.concat(intervals, sort=False).reset_index(drop=True)
        return intervals

    def add_wells(self, wellnames, renames=None, purposes=None,
                  gasfactors=None, readlogs=True, computetvdss=True,
                  cleanup=False, verb=False):
        """Add wells to project

        Parameters
        ----------
        wellnames : :obj:`list` or :obj:`tuple`
            Name of wells (as present in input files
            for Logs, Trajectories and Picks)'
        renames : :obj:`list` or :obj:`tuple`, optional
            Name of wells given within the pysubsurface project (and used to match
            picks in Picks file and object)
        purposes : :obj:`list` or :obj:`str`, optional
            Purpose of wells (can be wildcat, production,
            observation, injection, or ``None``)
        gasfactors : :obj:`list` or :obj:`float`, optional
            Factors to convert gas to reservoir condition (if ``None`` keep in
            surface conditions)
        readlogs : :obj:`bool`, optional
            Read welllogs from file (``True``) or not (``False``)
        computetvdss : :obj:`bool`, optional
            Compute TVDSS in depth axis of logs and add it to log curves
            (``True``) or not (``False``).
        cleanup : :obj:`bool`, optional
            Remove wells without picks from project (``True``) or
            keep all (``False``)
        verb : :obj:`bool`, optional
            Verbosity

        """
        renames = [None]*len(wellnames) if renames is None else renames
        if purposes is None:
            purposes = [None] * len(wellnames)
        elif isinstance(purposes, str):
            purposes = [purposes] * len(wellnames)
        if gasfactors is None:
            gasfactors = [None] * len(wellnames)
        elif isinstance(gasfactors, float):
            gasfactors = [gasfactors] * len(wellnames)

        for iwell, (wellname, rename, purpose, gasfactor) in \
                enumerate(zip(wellnames, renames, purposes, gasfactors)):
            if verb:
                print('Loading well {} '
                      'with name {} '
                      '(purpose = {})'.format(wellname,
                                              rename if rename is not None else
                                              wellname, purpose))
            well = Well(self.projectdir, wellname, rename, purpose=purpose,
                        readlogs=readlogs, gas_to_res=gasfactor,
                        kind=self._kind)
            if well.trajectory.df is not None:
                self.wells[rename] = well

                # add picks
                self.wells[rename].add_picks(self.picks,
                                             computetvdss=True,
                                             step_md=0.01)
                # create empty contacts
                self.wells[rename].create_contacts()

                # add intervals
                if len(self.wells[rename].picks.df) > 0:
                    self.wells[rename].create_intervals(self.intervals)
                else:
                    self.wells[rename].intervals = None
                # add TVDSS to logs
                if computetvdss and self.wells[rename].welllogs is not None:
                    self.wells[rename].compute_logs_tvdss()
            else:
                logging.warning('Cannot find trajectory for well {} in {}, '
                                'well will not be added...'.format(wellname,
                                                              self._kind))
        # remove wells without picks
        if cleanup:
            wellnames = list(self.wells.keys())
            for wellname in wellnames:
                if len(self.wells[wellname].picks.df) == 0:
                    self.wells.pop(wellname)
                    print('Removed well {}...'.format(wellname))
                    logging.warning('Removed well {}...'.format(wellname))

    def add_tdcurve(self, wellname, filename, tdname=None,
                    checkshot=True, computepicks=True,
                    computeintervals=True, computelog=True, verb=False):
        """Add TD curve (or checkshots) to a specific well

        Parameters
        ----------
        wellname : :obj:`str`
            Name of well
        filename : :obj:`str`
            Name of file containing TD curve or checkshots to be read
            (without extension)
        tdname : :obj:`str`, optional
            Name to give to TD curve or checkshots in the project (if ``None``
            use filename)
        checkshot : :obj:`bool`, optional
            curve is checkshot (``True``) or TD curve (``False``)
        computepicks : :obj:`bool`, optional
            Compute TWT for picks (``True``) or not (``False``).
        computeintervals : :obj:`bool`, optional
            Compute TWT for picks (``True``) or not (``False``). Can only be
            used in combination with ``computepicks=True``
        computelog : :obj:`bool`, optional
            Add TWT as a welllog curve in :class:``pysubsurface.objects.Logs` object
            or not (``False``).
        verb : :obj:`bool`, optional
            Verbosity

        """
        wellnames = list(self.wells.keys())
        if wellname in wellnames:
            if verb:
                print('Adding {}={} '
                      'for well {}'.format('checkshot' if checkshot else 'TDcurve',
                                           tdname, wellname))
            self.wells[wellname].add_tdcurve(filename, name=tdname,
                                             checkshot=checkshot)
            if computepicks:
                if checkshot:
                    self.wells[wellname].compute_picks_twt(checkshot_name=tdname)
                    if len(self.wells[wellname].contacts.df) > 0:
                        self.wells[wellname].compute_picks_twt(checkshot_name=tdname, contacts=True)
                else:
                    self.wells[wellname].compute_picks_twt(tdcurve_name=tdname)
                    if len(self.wells[wellname].contacts.df) > 0:
                        self.wells[wellname].compute_picks_twt(tdcurve_name=tdname, contacts=True)
                if computeintervals:
                    self.wells[wellname].add_intervals_twt(
                        twt_curve='TWT - {} (ms)'.format(tdname))
            if computelog:
                if checkshot:
                    self.wells[wellname].compute_logs_twt(checkshot_name=tdname)
                else:
                    self.wells[wellname].compute_logs_twt(tdcurve_name=tdname)

        else:
            raise ValueError('{} not in list of '
                             'available wells'.format(wellname))

    def add_polygon(self, filename, polygonname, verb=False):
        """Add polygon

        Parameters
        ----------
        filenames : :obj:`list` or :obj:`tuple`
           Name of file containing polygon (without extension)
        polygonname : :obj:`str`, optional
            Name of polygon (if ``None`` use filename)
        verb : :obj:`bool`, optional
           Verbosity

        """
        filename = os.path.join(self.projectdir, 'Polygon', filename + '.dat')
        self.polygons[polygonname] = Polygon(filename,
                                             polygonname=polygonname,
                                             kind=self._kind,
                                             verb=verb)

    def add_polygonset(self, filename, polygonname, verb=False):
        """Add polygon set

        Parameters
        ----------
        filenames : :obj:`list` or :obj:`tuple`
           Name of file containing polygon set (without extension)
        polygonname : :obj:`str`, optional
            Name of polygon set (if ``None`` use filename)
        verb : :obj:`bool`, optional
           Verbosity

        """
        filename = os.path.join(self.projectdir, 'Polygon', filename + '.dat')
        self.polygonsets[polygonname] = PolygonSet(filename,
                                                   polygonname=polygonname,
                                                   kind=self._kind,
                                                   verb=verb)

    def add_horizonset(self, setname, filenames, names, colors,
                       format='dsg5_long', domain='time', survey=None,
                       intervals=None, verb=False, **kwargs_interp):
        """Add set of horizons representing an interpretation

        Parameters
        ----------
        setname : :obj:`str`
            Name of horizon set
        filenames : :obj:`list` or :obj:`tuple`
            Name of files containing horizons (without extension)
        names : :obj:`list` or :obj:`tuple`
            Name to give to horizons (if ``None`` use filenames)
        colors : :obj:`list` or :obj:`tuple`
            Colors to assign to horizons and uses when displaying them
            together with seismic data
        format : :obj:`str`, optional
            Format of files containing surface (available options: ``dsg5_long``
            and ``plain_long``)
        domain : :obj:`str`, optional
            Domain of horizons (``time`` or ``depth``)
        survey : :obj:`str`, optional
            Survey name
        intervals : :obj:`tuple`, optional
            Names of intervals associated to horizons (must be N-1 where N is
            the number of horizons and in agreement with the
            :class:`pysubsurface.objects.Interval` object for the project)
        verb : :obj:`bool`, optional
            Verbosity
        kwargs_interp : :obj:`dict`, optional
            Additional parameters to be passed to
            :class:`pysubsurface.objects.Interpretation` initialization

        """
        if domain == 'depth':
            horsdir = os.path.join(self.projectdir, 'Surface',
                                   'Horizon', 'Depth')
        elif domain == 'time':
            horsdir = os.path.join(self.projectdir, 'Surface',
                                   'Horizon', 'Time')

        else:
            raise ValueError('Domain must be either time or depth...')
        if filenames is not None:
            filenames = [os.path.join(horsdir, hor) for hor in filenames]
        else:
            filenames = []
        self.horizonsets[setname+'-'+domain] = \
            {'domain': domain,
             'survey': survey,
             'names': list(names) if names is not None else [],
             'colors': list(colors) if colors is not None else [],
             'intervalnames': intervals,
             'data': Interpretation(filenames=filenames,
                                    format=format,
                                    kind=self._kind,
                                    verb=verb,
                                    **kwargs_interp)}

    def add_horizon_to_horizonset(self, setname, name, color, horizon,
                                  index=-1):
        """Add horizon to horizon set

        A constant horizon is added to an horizon set. However, being a fluid
        contact, the ``top`` reservoir horizon is used to mask the contact
        such that it will only contain values for :math:`z > z_{top}`

        Parameters
        ----------
        setname : :obj:`str`
            Name of horizon set
        name : :obj:`list` or :obj:`tuple`
            Name to give to horizon
        color : :obj:`list` or :obj:`tuple`
            Colors to assign to horizon and uses when displaying them
            together with seismic data
        horizon : :obj:`pysubsurface.object.Surface`
            Horizon to add
        index : :obj:`int`, optional
            Index of position in list where horizon will be added

        """
        if index == -1:
            index = len(self.horizonsets[setname]['names'])
        self.horizonsets[setname]['names'].insert(index, name)
        self.horizonsets[setname]['colors'].insert(index, color)
        self.horizonsets[setname]['data'].add_surface(horizon,
                                                      name=name,
                                                      index=index)

    def add_constanthorizon_to_horizonset(self, setname, name, color,
                                          constant, copyfrom=0):
        """Add constant horizon to horizon set

        Parameters
        ----------
        setname : :obj:`str`
            Name of horizon set
        name : :obj:`list` or :obj:`tuple`
            Name to give to horizon
        color : :obj:`list` or :obj:`tuple`
            Colors to assign to horizon and uses when displaying them
            together with seismic data
        constant : :obj:`float`
            Constant value for surface
        copyfrom : :obj:`int`, optional
            Index of surface to be used as template for new one

        """
        self.horizonsets[setname]['names'].append(name)
        self.horizonsets[setname]['colors'].append(color)
        self.horizonsets[setname]['data'].add_constant_surface(constant,
                                                               copyfrom, name=name)

    def add_fluidcontact_to_horizonset(self, setname, name, color,
                                       constant, top):
        """Add fluid contact horizon to horizon set

        A constant horizon is added to an horizon set. However, being a fluid
        contact, the ``top`` reservoir horizon is used to mask the contact
        such that it will only contain values for :math:`z > z_{top}`

        Parameters
        ----------
        setname : :obj:`str`
            Name of horizon set
        name : :obj:`list` or :obj:`tuple`
            Name to give to horizon
        color : :obj:`list` or :obj:`tuple`
            Colors to assign to horizon and uses when displaying them
            together with seismic data
        constant : :obj:`float`
            Constant value for surface
        top : :obj:`str`
            Name of horizon to be used as top

        """
        self.horizonsets[setname]['names'].append(name)
        self.horizonsets[setname]['colors'].append(color)
        self.horizonsets[setname]['data'].add_fluidcontact(constant,
                                                           self.horizonsets[setname]['names'].index(top),
                                                           name=name)

    def add_fault(self, filename, faultname, color, domain='time',
                  survey=None, verb=False):
        """Add fault

        Parameters
        ----------
        setname : :obj:`str`
            Name of fault
        filename : :obj:`list` or :obj:`tuple`
            Name of files containing horizons (without extension)
        names : :obj:`list` or :obj:`tuple`
            Name to give to horizons (if ``None`` use filenames)
        color : :obj:`list` or :obj:`tuple`
            Color to assign to fault and uses when displaying them
            together with seismic data
        domain : :obj:`str`, optional
            Domain of fault (``time`` or ``depth``)
        survey : :obj:`str`, optional
            Survey name
        verb : :obj:`bool`, optional
            Verbosity

        """
        if domain == 'depth':
            faultdir = os.path.join(self.projectdir, 'Fault', 'Depth')
        elif domain == 'time':
            faultdir = os.path.join(self.projectdir, 'Fault', 'Time')
        else:
            raise ValueError('Domain must be either time or depth...')
        filename = os.path.join(faultdir, filename + '.dat')

        self.faults[faultname+'-'+domain] = \
            {'domain': domain,
             'survey': survey,
             'color': color,
             'data': Fault(filename=filename,
                           kind=self._kind,
                           verb=verb)}

    def add_seismic(self, filename, name, type='Post',
                    domain='time', survey=None,
                    iline=189, xline=193, offset=37,
                    cdpy=185, cdpx=181,
                    scale=1, loadcube=False):
        """Add seismic data

        Parameters
        ----------
        filename : :obj:`str`
            Name of file containing seismic data (without extension)
        name : :obj:`str`, optional
            Name to give to seismic data (if ``None`` use filename)
        type : :obj:`str`, optional
            Type of data: ``post``:Post-stack or ``pre``:Pre-stack
        domain : :obj:`str`, optional
            Domain of seismic (``time`` or ``depth``)
        survey : :obj:`str`, optional
            Survey name
        iline : :obj:`int`, optional
            Byte location of inline
        xline : :obj:`int`, optional
            Byte location of crossline
        offset : :obj:`int`, optional
            Byte location of offset
        cdpy : :obj:`int`, optional
            Byte location of CDP_Y
        cdpx : :obj:`int`, optional
            Byte location of CDP_X
        scale : :obj:`float`, optional
            Apply scaling to data
        loadcube : :obj:`bool`, optional
            Read data during initialization (``True``) or not (``False``)

        """
        filename = os.path.join(self.projectdir, 'Seismic', type,
                                filename + '.sgy')
        if type == 'Post' and domain in ['depth', 'time']:

            self.seismics[name] = {'type': type,
                                   'domain': domain,
                                   'survey': survey}

            try:
                self.seismics[name]['data'] = Seismic(filename=filename,
                                                      iline=iline, xline=xline,
                                                      cdpy=cdpy, cdpx=cdpx,
                                                      loadcube=loadcube,
                                                      taxis=True if
                                                      domain == 'time' else
                                                      False, scale=scale,
                                                      kind=self._kind,)
            except Exception as e:
                self.seismics[name]['data'] = SeismicIrregular(filename=filename,
                                                               iline=iline,
                                                               xline=xline,
                                                               cdpy=cdpy,
                                                               cdpx=cdpx,
                                                               loadcube=loadcube,
                                                               taxis=True if
                                                               domain == 'time' else
                                                               False, scale=scale,
                                                               kind=self._kind,)
        elif type == 'Pre' and domain in ['depth', 'time']:

            self.preseismics[name] = {'type': type,
                                      'domain': domain,
                                      'survey': survey}

            self.preseismics[name]['data'] = PrestackSeismic(filename=filename,
                                                             iline=iline,
                                                             xline=xline,
                                                             offset=offset,
                                                             cdpy=cdpy,
                                                             cdpx=cdpx,
                                                             loadcube=loadcube,
                                                             taxis=True if
                                                             domain == 'time' else
                                                             False, scale=scale,
                                                             kind=self._kind,)

        else:
            raise ValueError('type={} must be either Post or Pre, '
                             'and domain={} must be either '
                             'time or depth'.format(type, domain))

    def add_facies(self, faciesname, color, intervalname=None,
                   level=None, filters=None):
        """Add facies definition to the project in the form of
         :class:`pysubsurface.objects.Facies` object

        Parameters
        ----------
        faciesname : :obj:`str`
            Name of facies
        color : :obj:`str`
            Color to assign to facies
        intervalname : :obj:`str` or :obj:`tuple`, optional
            Name of interval
             (if ``None`` consider entire depth range and do not
             filter on interval)
        level : :obj:`str`
            Level of interval(s)
        filters : :obj:`list` or :obj:`tuple`
            Filters to be applied
            (each filter is a dictionary with logname and rule, e.g.
            logname='LFP_COAL', rule='<0.1' will keep all values where values
            in  LFP_COAL logs are <0.1)
        """
        self.facies[faciesname] = Facies(faciesname, color,
                                         intervalname=intervalname,
                                         level=level,
                                         filters=filters)

    def copy_well(self, well, wellname):
        """Copy a well into the project

        Parameters
        ----------
        well : :obj:`pysubsurface.objects.Well`
            Well object
        wellname : :obj:`str`
            Name of well

        """
        self.wells[wellname] = well

    def copy_horizonset(self, horizonset, setname):
        """Copy an horizonset into the project

        Parameters
        ----------
        horizonset : :obj:`dict`
            Dictionary containing an horizonset with the following fields:
            domain, survey, names, colors, intervalnames, data
        setname : :obj:`str`
            Name of horizon set

        """
        self.horizonsets[setname] = copy.deepcopy(horizonset)

    @deprecated(version='v0.0.0', reason="Use extract_wells_subset")
    def get_wellsubset(self, wellnames):
        """Return a subset of wells

        Parameters
        ----------
        wellnames : :obj:`tuple` or :obj:`str`
            Name of wells to extract

        Returns
        -------
        wellsubset : :obj:`dict`
            Subset of wells

        """
        wellsubset = {wellrename: self.wells[wellrename]
                      for wellrename in wellnames}
        return wellsubset

    def extract_wells_subset(self, wellnames=None, regexp=None):
        """Return a subset of wells

        Parameters
        ----------
        wellnames : :obj:`list`, optional
            Well names
        regexp : :obj:`list`, optional
            Regular expression to select wells (used if wellnames is ``None``)

        Returns
        -------
        wells : :obj:`dict`
            Subset of wells

        """
        if wellnames is None:
            r = re.compile(regexp)
            wellnames = list(filter(r.match, list(self.wells.keys())))

        wells = {}
        for wellname in wellnames:
            if wellname in self.wells.keys():
                wells[wellname] = self.wells[wellname]
        return wells

    def return_surface_by_name(self, horizonsetname, name):
        """Return surface by name.

        Return surface at position of ``name`` in horizonset.
        If ``name`` is not present in the interpretation set, ``None``
        is returned.

        Parameters
        ----------
        horizonsetname : :obj:`str`
            Name of horizon set
        name : :obj:`str`
            Name of surface to be returned

        Returns
        -------
        surface : :obj:`pysubsurface.objects.Surface`
            Surface

        """
        isurface = self.horizonsets[horizonsetname]['names'].index(name)
        surface = None
        if isurface is not None:
            surface = self.horizonsets[horizonsetname]['data'].surfaces[isurface]
        return surface

    def assign_color_picks(self):
        """Automatically color all picks based on color in Top field of
        intervals member of the class.
        """
        # single field (deprecated)
        #allpicks = self.picks.df['Name'].unique()
        #for pick in allpicks:
        #    interval = self.intervals.df[self.intervals.df['Top'] == pick]
        #    if len(interval) > 0:
        #        self.picks.df['Color'][self.picks.df['Name'] == pick] = \
        #            interval.iloc[0]['Color']

        fields = self.picks.df['Field'].unique()
        for field in fields:
            picks_field = self.picks.df[self.picks.df['Field'] == field]
            allpicks = picks_field['Name'].unique()
            for pick in allpicks:
                interval = self.intervals.df[self.intervals.df['Top'] == pick]
                if len(interval) > 0:
                    self.picks.df['Color'][(self.picks.df['Name'] == pick) & (self.picks.df['Field'] == field)] = \
                        interval.iloc[0]['Color']

    def interpret_horizonsetintervals(self, setname, level):
        """Map ``intervalnames`` from an horizonset named ``setname``
        to actual intervals using the ``intervals`` internal object for the
        mapping

        Parameters
        ----------
        setname : :obj:`str`
            Name of horizon set
        level : :obj:`int`
            Interval level

        """
        if level is None:
            intervals = self.intervals.df[self.intervals.df['Level'].isnull()]
        else:
            intervals = self.intervals.df[self.intervals.df['Level'] == level]

        self.horizonsets[setname]['intervals'] = \
            intervals[intervals['Name'] == self.horizonsets[setname]['intervalnames'][0]]

        for intervalname in self.horizonsets[setname]['intervalnames'][1:]:
            self.horizonsets[setname]['intervals'] = \
                self.horizonsets[setname]['intervals'].append(intervals[intervals['Name'] ==
                                                                        intervalname])

    def return_facies_colors(self):
        """Return color for all available facies

        Returns
        -------
        colors : :obj:`list`
            Colors

        """
        facies = list(self.facies.keys())
        colors = [self.facies[fac].color for fac in facies]
        return colors

    def assign_facies_to_wells(self):
        """Assign facies mapping to all wells and write a new welllog with
        facies definition.

        """
        wellnames = self.wells.keys()
        for wellname in wellnames:
            self.wells[wellname].assign_facies(self.facies)

    def intervals_stats(self, level=None, plotflag=True):
        """Statistics on intervals for all wells

        Parameters
        ----------
        level : :obj:`int` or :obj:`list`, optional
            Level(s) to analyze (provide two at least). If ``None``, analyse all
        plotflag : :obj:`bool`
            Plot boxplot (``True``) or print summary (``False``)

        Returns
        -------
        intervals : :obj:`pd.DataFrame`
            Intervals for all wells
        intervals_stats : :obj:`pd.DataFrame`
            Intervals statistics

        """
        # collect all intervals from wells
        intervals = self._collect_intervals()

        # remove intervals with zero thickness
        intervals = intervals[intervals['Thickness (meters)'] > 0]

        # switch None to -1
        intervals['Level'] = intervals['Level'].fillna(value=-1)

        # extract only levels of interest
        if level is not None:
            intervals = intervals[intervals['Level'].isin(level)]

        # make statistics for each interval
        intervals_stats = \
            intervals.groupby(by='Name').agg({'Thickness (meters)': ['mean',
                                                                     'std'],
                                              'Level': ['min'],
                                              'Color': ['min']})

        # display interval statistics
        if ipython_flag:
            display(intervals_stats.style.applymap(lambda x:
                                                   'background-color: %s' % x,
                                                   subset=['Color']))
        else:
            print(intervals_stats())

        # plot interval statistics
        if plotflag:
            avail_levels = intervals['Level'].unique()

            # box plot of thicknesses per interval
            fig1, axs1 = plt.subplots(len(avail_levels), 1,
                                      figsize=(12, 5 * len(avail_levels)))
            if len(avail_levels) == 1:
                axs1 = [axs1, None]
            for iax, level in enumerate(avail_levels):
                intervals_level = intervals[intervals['Level'] == level]
                axs1[iax] = sns.boxplot(y="Name", x="Thickness (meters)",
                                        data=intervals_level,
                                        palette=intervals_level['Color'].unique(),
                                        ax=axs1[iax])
                axs1[iax] = sns.stripplot(y="Name", x="Thickness (meters)",
                                          data=intervals_level, jitter=0,
                                          color='k', ax=axs1[iax])
                axs1[iax].set_title('Statistics '
                                    'for level {}'.format('None' if level == -1 else level),
                                    weight='bold')
            plt.tight_layout()

            # distribution plot of thicknesses per interval
            fig2, axs2 = plt.subplots(len(avail_levels), 1,
                                      figsize=(10, 5 * len(avail_levels)))
            if len(avail_levels) == 1:
                axs2 = [axs2, None]
            for iax, level in enumerate(avail_levels):
                intervals_level = intervals[intervals['Level'] == level]
                intnames = intervals_level['Name'].unique()

                for intname in intnames:
                    interval = intervals_level[intervals_level['Name'] == intname]
                    if len(interval) > 1:
                        sns.distplot(interval['Thickness (meters)'], hist=False,
                                     rug=True,
                                     rug_kws={'color': interval.iloc[0]['Color'],
                                              'lw': 5, 'height': 0.1},
                                     kde_kws={'color': interval.iloc[0]['Color'],
                                              'lw': 3, 'shade': True, 'alpha': 0.6,
                                              'label': intname}, ax=axs2[iax])
            plt.tight_layout()

        if plotflag:
            return intervals, intervals_stats, fig1, axs1, fig2, axs2
        else:
            return intervals, intervals_stats, None, None, None, None

    def create_averageprops_intervals(self, level=2, vpname='LFP_VP',
                                      vsname='LFP_VS', rhoname='LFP_RHOB',
                                      ainame=None, vpvsname=None,
                                      filters=None, avename=None):
        """Compute statistics for elastic properties
        within intervals of all wells, save in self.welllogs object, and
        return well-averaged statistics

        Parameters
        ----------
        level : :obj:`str`, optional
            Level to analyze
        vpname : :obj:`str`, optional
            Name of log containing P-wave velocity
        vsname : :obj:`str`, optional
            Name of log containing S-wave velocity
        rhoname : :obj:`str`, optional
            Name of log containing density
        ainame : :obj:`str`, optional
            Name of log containing acoustic impedence
        vpvsname : :obj:`str`, optional
            Name of log containing vpvs velocity ratio
        filters : :obj:`tuple`, optional
            Filters to be applied during extraction
            (each filter is a dictionary with logname, rule and chain, e.g.
            ``logname='LFP_COAL'``, ``rule='<0.1'``, ``chain='and'/'or'``
            will keep all values where values in LFP_COAL logs are <0.1 and
            will be combined with additional rules with an and/or conditional)
            Note that chaining rule is strictly not needed for the first filter
        avename : :obj:`str`, optional
            Name to give to average properties saved in ``aveprops`` member of
            Project object (if ``None`` do not store in the object, simply
            return it)

        Returns
        -------
        aveprops : :obj:`dict` or :obj:`pd.DataFrame`
            Wells-averaged statistics for every interval in chosen level

        """
        def _average_props(props, weights):
            """Average mean and stdev among different dataframes using
            weights based on number of samples used to produce mean and stdev
            for each dataframe
            """
            props = pd.concat(props).T
            props_mean = props.drop(columns='stdev')
            props_stdev = props.drop(columns='mean')
            weights.columns = props_mean.columns
            props_mean = (props_mean*weights).sum(axis=1)
            weights.columns = props_stdev.columns
            props_stdev = (props_stdev*weights).sum(axis=1)
            props_stats = pd.concat((props_mean, props_stdev), axis=1)
            props_stats.columns = ['mean', 'stdev']
            props_stats = props_stats.T.to_dict()
            return props_stats

        # retrieve averaged properties for each well
        wellnames = list(self.wells.keys())
        aveprops_wells = {}
        for wellname in wellnames:
            aveprops_wells[wellname] = \
                self.wells[wellname].create_averageprops_intervals(
                    level=level, vpname=vpname,
                    vsname=vsname, rhoname=rhoname, ainame=ainame,
                    vpvsname=vpvsname, filters=filters)

        # reorganize properties for all wells
        aveprops_vp = [pd.DataFrame(aveprops_wells[wellnames[0]][vpname])]
        aveprops_vs = [pd.DataFrame(aveprops_wells[wellnames[0]][vsname])]
        aveprops_rho = [pd.DataFrame(aveprops_wells[wellnames[0]][rhoname])]
        aveprops_ai = [pd.DataFrame(aveprops_wells[wellnames[0]][ainame])]
        aveprops_vpvs = [pd.DataFrame(aveprops_wells[wellnames[0]][vpvsname])]
        aveprops_cov = [aveprops_wells[wellnames[0]]['Cov']]
        aveprops_nsamples = [aveprops_wells[wellnames[0]]['Nsamples']]

        for wellname in wellnames[1:]:
            aveprops_vp.append(pd.DataFrame(aveprops_wells[wellname][vpname]))
            aveprops_vs.append(pd.DataFrame(aveprops_wells[wellname][vsname]))
            aveprops_rho.append(pd.DataFrame(aveprops_wells[wellname][rhoname]))
            aveprops_ai.append(pd.DataFrame(aveprops_wells[wellname][ainame]))
            aveprops_vpvs.append(pd.DataFrame(aveprops_wells[wellname][vpvsname]))
            aveprops_cov.append(aveprops_wells[wellname]['Cov'])
            aveprops_nsamples.append(aveprops_wells[wellname]['Nsamples'])

        aveprops_nsamples = pd.DataFrame(aveprops_nsamples).T
        aveprops_weights = \
            aveprops_nsamples.apply(lambda x: x / aveprops_nsamples.sum(axis=1),
                                    axis=0).fillna(0)

        # average mean and stdev
        aveprops_vp = _average_props(aveprops_vp, aveprops_weights)
        aveprops_vs = _average_props(aveprops_vs, aveprops_weights)
        aveprops_rho = _average_props(aveprops_rho, aveprops_weights)
        aveprops_ai = _average_props(aveprops_ai, aveprops_weights)
        aveprops_vpvs = _average_props(aveprops_vpvs, aveprops_weights)

        # average covariances
        aveprops_covall = {}
        for interval in aveprops_cov[0].keys():
            aveprops_cov_interval = \
                [aveprops[interval].values for aveprops in aveprops_cov \
                 if interval in aveprops.keys()]
            iava = [i for i, aveprops in enumerate(aveprops_cov) \
                 if interval in aveprops.keys()]
            aveprops_nsamples_interval = aveprops_nsamples.loc[interval].T.iloc[iava]
            aveprops_covall[interval] = \
                average_stats(aveprops_cov_interval, aveprops_nsamples_interval)

        # return averaged props in dictiorary
        aveprops = {vpname: aveprops_vp,
                    vsname: aveprops_vs,
                    rhoname: aveprops_rho,
                    ainame: aveprops_ai,
                    vpvsname: aveprops_vpvs,
                    'Cov': aveprops_covall}
        if avename is not None:
            self.aveprops[avename] = {'filters': filter, 'aveprops': aveprops}
        return aveprops

    def mapdocfigs(self, verb=False):
        """Map figures from local directory to documentation directory

        Parameters
        ----------
        verb : :obj:`bool`, optional
            Verbosity

        """
        if self.docfigdir is not None:
            if verb:
                print('Copying figures from {} to {}'.format(self.figdir,
                                                             self.docfigdir))
            copy_tree(self.figdir, self.docfigdir)

    #########
    # Viewers
    #########
    def view_polygons(self, ax, color='k', flipaxis=False,
                      bbox=False, axiskm=False):
        """Visualize polygons on ``ax`` axis

        Parameters
        ----------
        ax : :obj:`plt.axes`
            Axes handle
        color : :obj:`str`, optional
            Trajectory color
        flipaxis : :obj:`bool`, optional
            Plot x coordinates on x-axis and y coordinates on y-axis (``False``)
            or viceversa (``True``)
        bbox : :obj:`bool`, optional
            Add box around labels
        axiskm : :obj:`bool`, optional
            Show axis in km units (``True``) or m units (``False``)

        Returns
        -------
        ax : :obj:`plt.axes`
            Axes handle

        """
        for polyname in self.polygons.keys():
            self.polygons[polyname].view(ax, color=color,
                                         polyname=polyname,
                                         flipaxis=flipaxis,
                                         bbox=bbox, axiskm=axiskm)
        return ax

    ##########################################################
    # Documentation viewers (with standardized figure filename
    ##########################################################
    def view_intervals(self, levelmax=None, field=None, govern_area=None):
        """Visualize stratigraphic column.

        The stratigrapic column of the project is visaulized. If the project
        contains multiple fields/govern_area, the first one will be visualized
        unless the name of the field of interest is provided. Note that this is
        a thin wrapper for end-users to the :func:'pysubsurface.objects.Interval.view'
        function.

        Parameters
        ----------
        levelmax : :obj:`int`, optional
            Maximum level to display
        field : :obj:`str`, optional
            Name of field to visualize
        govern_area : :obj:`str`, optional
            Name of govern area to visualize. If ``None``, the first govern area
            in alphabetic order will be selected


        """
        figpath = _create_figpath(self.figdir, 'geology')
        title = self.projectname+' Stratigraphy' if field is None else field+' Stratigraphy'
        _, _ = \
            self.intervals.view(levelmax=levelmax, field=field,
                                govern_area=govern_area, alpha=0.7,
                                figsize=(10, 15), fontsize=9,
                                title=title,
                                savefig=os.path.join(figpath,
                                                     self._projectname_nospace+
                                                     '_Stratigraphy.png'))

    def view_count_picks(self):
        """Visualize picks count for entire field. This is a thin wrapper
        for end-users to the :func:'pysubsurface.objects.Picks.count_picks' function.

        """
        figpath = _create_figpath(self.figdir, 'geology')

        self.picks.count_picks(nrows=20, plotflag=True,
                               title=self.projectname+ ' Picks count',
                               savefig=os.path.join(figpath,
                                                    self._projectname_nospace +
                                                    '_PicksCount.png'))

    def view_logtrack(self, wellname, savefig=True, **kwargs):
        """Display log track using any of the provided standard templates
        tailored to different disciplines and analysis

        This is a thin wrapper around :func:`pysubsurface.objects.Well.view_logtrack`,
        refer to the documentation of :func:`pysubsurface.objects.Well.view_logtrack`
        for more details.

        Parameters
        ----------
        wellname : :obj:`str`
            Name of well to display
        savefig : :obj:`str`, optional
            Save figure (``True``) or not (``False``)
        kwargs : :obj:`dict`, optional
            input parameters to be provided to
            :func:`pysubsurface.objects.Well.view_logtrack`

        Returns
        -------
        fig : :obj:`plt.figure`
           Figure handle (``None`` if ``axs`` are passed by user)
        axs : :obj:`plt.axes`
           Axes handles

        """
        fig, axs = self.wells[wellname].view_logtrack(**kwargs)
        fig.subplots_adjust(right=0.9)

        # savefig
        if savefig:
            template = 'petro' if 'template' not in kwargs else kwargs['template']
            figpath = _create_figpath(self.figdir, 'petrophysics' if template == 'petro' else 'geophysics')
            fig.savefig(os.path.join(figpath, self._projectname_nospace+'_'+
                                     change_name_for_unix(wellname)+'_'+
                                     template+'logs.png'),
                        dpi=300, bbox_inches='tight')
        return fig, axs

    def view_logprops_wells(self, level=2, wells=None, interval=None,
                            prop1name='VP', prop2name=None, filters=None,
                            bins=None, cmap='rainbow', xaxisrev=False,
                            yaxisrev=False, xlim=None, ylim=None,
                            combinedhist=False, fit=False,
                            figsize=(10, 5), title=None, figsubdir=None,
                            savefig=True):
        """Histograms or scatter plots of log properties grouped
        by well

        Filter and display log properties grouped by well in histogram or
        scatter plot. A filter can be applied to select wells, a specific level
        or a specific interval.

        Parameters
        ----------
        level : :obj:`int`, optional
            Interval level to display
        wells : :obj:`str` or:obj:`tuple`, optional
            Well or wells to display (if ``None`` visualize all
            available wells)
        interval : :obj:`str` , optional
            Interval  to display (if ``None`` used entire log)
        prop1name : :obj:`str` , optional
            Property to visualize (use name in :class:`pysubsurface.objects.Logs`
            objects)
        prop2name : :obj:`str` , optional
            Second property to visualize (if ``None`` display histograms, if not
            ``None`` display scatterplots)
        filters : :obj:`tuple` , optional
            Filters to be applied during extraction
            (each filter is a dictionary with logname and rule, e.g.
            logname='LFP_COAL', rule='<0.1' will keep all values where values
            in  LFP_COAL logs are <0.1)
        bins : :obj:`np.ndarray` , optional
            Bins for histogram (if ``None`` infer from data)
        cmap : :obj:`str`, optional
            Colormap
        xaxisrev : :obj:`bool`, optional
            Reverse x-axis
        yaxisrev : :obj:`bool`, optional
            Reverse y-axis
        xlim : :obj:`tuple`, optional
            x-axis limits (if ``None`` infer from data)
        ylim : :obj:`tuple`, optional
            y-axis limits (if ``None`` infer from data)
        combinedhist : :obj:`bool`, optional
            Draw one histogram per well (``False``) or an histogram for all
            wells (``True``)
        fit : :obj:`bool`, optional
            Compute and draw best-fit line in scatterplot
        figsize : :obj:`tuple`, optional
            Size of figure
        title : :obj:`str`, optional
            Title of figure
        figsubdir : :obj:`str`, optional
            Subdirectory where figure will be saved
        savefig : :obj:`str`, optional
            Figure filename without extension.
            Path and prefix will be added automatically from info in project
            (if ``None``, figure is not saved)

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle (``None`` if ``axs`` are passed by user)
        axs : :obj:`plt.axes`
            Axes handles

        """
        # find out wells to use (and remove those without logs)
        wellnames = list(self.wells.keys()) if wells is None else wells
        wellnames = [wellname for wellname in wellnames if self.wells[wellname].welllogs is not None]
        nwells = len(wellnames)

        combinedlog1 = []
        if prop2name:
            combinedlog2=[]

        # create colormap
        cmap = _discrete_cmap(nwells, cmap)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.grid()
        # collect data (and scatter plot) for all wells
        for iwell, wellname in enumerate(wellnames):
            if interval:
                selinterval = \
                    self.wells[wellname].intervals[(self.wells[wellname].intervals['Name'] == interval) &
                                                   (self.wells[wellname].intervals['Level'] == level)]
                if len(selinterval)==1:
                    logcurve1 = \
                        self.wells[wellname].extract_logs_in_interval(selinterval.iloc[0],
                                                                      prop1name,
                                                                      filters=filters)
                else:
                    logcurve1 = np.empty(0)
            else:
                logcurve1 = self.wells[wellname].welllogs.logs[prop1name]
            combinedlog1.append(logcurve1)
            if prop2name:
                if interval is None:
                    logcurve2 = self.wells[wellname].welllogs.logs[prop2name]
                else:
                    if len(selinterval) == 1:
                        logcurve2 = self.wells[wellname].extract_logs_in_interval(selinterval.iloc[0],
                                                                                  prop2name, filters=filters)
                    else:
                        logcurve2 = np.empty(0)
                combinedlog2.append(logcurve2)

                if logcurve1 is not None and logcurve2 is not None:
                    im = ax.scatter(logcurve1, logcurve2,
                                    c=iwell*np.ones(len(logcurve1)),
                                    marker='o', edgecolors='none',
                                    alpha=0.7, cmap=cmap,
                                    vmin=0, vmax=nwells)
            else:
                if not combinedhist and len(logcurve1[~np.isnan(logcurve1)])>0:
                    sns.distplot(logcurve1[~np.isnan(logcurve1)],
                                 fit=norm, rug=False, bins=bins,
                                 hist_kws={'color': cmap(iwell), 'alpha': 0.5},
                                 kde_kws={'color': cmap(iwell), 'lw': 3,
                                          'label': wellname},
                                 fit_kws={'color': cmap(iwell),
                                          'lw': 3, 'ls': '--'},
                                 ax=ax)

        # histogram plot data for all wells
        if not prop2name and combinedhist:
            combinedlog1 = np.concatenate(combinedlog1)
            combinedlog1 = combinedlog1[~np.isnan(combinedlog1)]
            sns.distplot(combinedlog1[~np.isnan(combinedlog1)],
                         fit=norm, rug=False, bins=bins,
                         hist_kws={'color': 'k', 'alpha': 0.5},
                         kde_kws={'color': 'k', 'lw': 3},
                         fit_kws={'color': 'k', 'lw': 3, 'ls': '--'},
                         ax=ax)
            if bins is not None: ax.set_xlim(bins[0], bins[-1])
            ax.text(0.95 * ax.get_xlim()[1], 0.85 * ax.get_ylim()[1],
                    'mean: %.3f,\n std: %.3f' % (np.mean(combinedlog1),
                                                 np.std(combinedlog1)),
                    fontsize=14, ha="right", va="center",
                    bbox=dict(boxstyle="square",
                              ec=(0., 0., 0.),
                              fc=(1., 1., 1.)))

        # best fit line
        if prop2name and fit:
            combinedlog1 = np.concatenate(combinedlog1)
            combinedlog2 = np.concatenate(combinedlog2)
            isnanlog = (np.isnan(combinedlog1)) | (np.isnan(combinedlog2))
            combinedlog1 = combinedlog1[~isnanlog].reshape(-1, 1)
            combinedlog2 = combinedlog2[~isnanlog].reshape(-1, 1)
            reg = LinearRegression().fit(combinedlog1, combinedlog2)
            ax.plot(np.array([combinedlog1.min(), combinedlog1.max()]),
                    np.array([combinedlog1.min(), combinedlog1.max()]) *
                    reg.coef_[0] + reg.intercept_, 'k', lw=4,
                    label='bestfit: {0} = {1:.2f}*{2} '
                          '{3} {4:.2f}'.format(prop2name, reg.coef_[0][0],
                                               prop1name,
                                               '+' if reg.intercept_> 0 else '-',
                                               abs(reg.intercept_[0])))

        if prop2name and fit:
            ax.legend()

        # limit and reverse axes, add colorbar if scatterplot is shown
        ax.set_xlabel(prop1name)
        if xlim: ax.set_xlim(xlim)
        if xaxisrev: ax.invert_xaxis()
        if prop2name:
            ax.set_ylabel(prop2name)
            if ylim: ax.set_ylim(ylim)
            if yaxisrev: ax.invert_yaxis()
            cbar = fig.colorbar(im, ax=ax,
                                ticks= np.arange(0, nwells + 1.) + 0.5)
            cbar.set_ticklabels(wellnames)
            cbar.ax.set_ylabel('Well', rotation=270, fontsize=12)
        if title: ax.set_title(title, weight='bold')
        plt.tight_layout()

        # savefig
        if savefig is not None and figsubdir is not None:
            figpath = _create_figpath(self.figdir, figsubdir)
            fig.savefig(os.path.join(figpath,
                                     self._projectname_nospace+'_'+savefig+'.png'),
                        dpi=300, bbox_inches='tight')
        return fig, ax

    def view_logprops_intervals(self, level, wells=None, intervals=None,
                                vpname='VP', vsname='VS', rhoname='RHOB',
                                prop1name='VP', prop2name='VS', filters=None,
                                draw=False, nsamples=1000,
                                xaxisrev=False, yaxisrev=False,
                                xlim=None, ylim=None,
                                figsize=(12, 5), title=None, savefig=None):
        """Histograms or scatter plots of log properties grouped
        by intervals

        Filter and display log properties grouped by well in histogram or
        scatter plot. A filter can be applied to select wells, a specific level
        or a specific interval.

        Parameters
        ----------
        level : :obj:`int`, optional
           Interval level to display
        wells : :obj:`str` or:obj:`tuple`, optional
           Well or wells to display (if ``None`` visualize all
           available wells)
        intervals : :obj:`str` or :obj:`tuple` , optional
           Interval or Intervals  to display (if ``None`` used entire log)
        vpname : :obj:`str` , optional
           Name of log containing average vp (needed if ``draw=True``)
        vsname : :obj:`str` , optional
           Name of log containing average vs (needed if ``draw=True``)
        rhoname : :obj:`str` , optional
           Name of log containing average density (needed if ``draw=True``)
        prop1name : :obj:`str` , optional
           Property to visualize (use name in :class:`pysubsurface.objects.Logs`
           objects)
        prop2name : :obj:`str` , optional
           Second property to visualize (if ``None`` display histograms, if not
           ``None`` display scatterplots)
        filters : :obj:`tuple`, optional
           Filters to be applied during extraction
           (each filter is a dictionary with logname and rule, e.g.
           logname='LFP_COAL', rule='<0.1' will keep all values where values
           in  LFP_COAL logs are <0.1)
        draw : :obj:`bool` , optional
           Draw samples (``True``) or use samples from well (``False``)
        nsamples : :obj:`int`, optional
           Number of samples to draw
        bins : :obj:`np.ndarray` , optional
           Bins for histogram (if ``None`` infer from data)
        cmap : :obj:`str`, optional
            Colormap
        xaxisrev : :obj:`bool`, optional
            Reverse x-axis
        yaxisrev : :obj:`bool`, optional
            Reverse y-axis
        xlim : :obj:`tuple`, optional
            x-axis limits (if ``None`` infer from data)
        ylim : :obj:`tuple`, optional
            y-axis limits (if ``None`` infer from data)
        figsize : :obj:`tuple`, optional
            Size of figure
        title : :obj:`str`, optional
            Title of figure
        savefig : :obj:`str`, optional
            Figure filename (if ``None``, figure is not saved)

        Returns
        -------
        fig : :obj:`plt.figure`
           Figure handle (``None`` if ``axs`` are passed by user)
        axs : :obj:`plt.axes`
           Axes handles

        """
        wellnames=list(self.wells.keys()) if wells is None else wells

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for iwell, wellname in enumerate(wellnames):
            _, ax = \
                self.wells[wellname].view_logprops_intervals(level=level,
                                                             ax=ax,
                                                             intervals=intervals,
                                                             vpname=vpname,
                                                             vsname=vsname,
                                                             rhoname=rhoname,
                                                             prop1name=prop1name,
                                                             prop2name=prop2name,
                                                             filters=filters,
                                                             draw=draw,
                                                             nsamples=nsamples,
                                                             xaxisrev=xaxisrev,
                                                             yaxisrev=yaxisrev,
                                                             xlim=xlim,
                                                             ylim=ylim,
                                                             legend=True if
                                                             iwell==0 else
                                                             False)
        if title: ax.set_title(title, weight='bold')
        # savefig
        if savefig is not None:
            fig.savefig(os.path.join(self.figdir, savefig),
                        dpi=300, bbox_inches='tight')
        return fig, ax


    def view_surface(self, surfacename, horizonset, title=None,
                     polygonset=None, addpolygons=True, addwells=True,
                     lims=None, savefig=True, **kwargs_surface):
        """Display surface (horizon or map) with wells and polygons

        Parameters
        ----------
        surfacename : :obj:`str`
            Name of surface
        horizonset : :obj:`str`
            Name of horizon set
        title : :obj:`str`, optional
            Figure title (if ``None`` use ``surfacename``)
        polygonset : :obj:`bool`, optional
            Name of polygonset to display
        addpolygons : :obj:`bool`, optional
            Add polygons
        addwells : :obj:`bool`, optional
            Add wells
        lims : :obj:`tuple`, optional
            y-x (or il-xl) limits
        savefig : :obj:`bool`, optional
            Save figure (``True``) or not (``False``)
        kwargs_surface : :obj:`dict`, optional
            Additional input parameters to be provided to
            :func:`pysubsurface.objects.Surface.view`

        Returns
        -------
        fig : :obj:`plt.figure`
           Figure handle (``None`` if ``axs`` are passed by user)
        ax : :obj:`plt.axes`
           Axes handle

        """
        surfacenames = self.horizonsets[horizonset]['names']

        try:
            isurface = surfacenames.index(surfacename)
        except:
            raise ValueError('{} not in {} set'.format(surfacename, horizonset))

        polygonset = self.polygonsets[polygonset] if polygonset is not None else None
        polygons = self.polygons if addpolygons else None
        wells = self.wells if addwells else None

        fig, ax = \
            self.horizonsets[horizonset]['data'].view_surface(isurface, title=title,
                                                              polygonset=polygonset,
                                                              polygons=polygons,
                                                              wells=wells,
                                                              lims=lims,
                                                              **kwargs_surface)

        # savefig
        if savefig:
            figpath = _create_figpath(self.figdir,
                                      os.path.join('geophysics', 'horizons'))
            fig.savefig(os.path.join(figpath, self._projectname_nospace + '_' +
                             surfacename + '.png'),
                        dpi=300, bbox_inches='tight',
                        facecolor=fig.get_facecolor(),
                        transparent=True)
        return fig, ax


    def view_faults(self, surfacename, horizonset, title=None,
                    addwells=True, lims=None, savefig=True, **kwargs_surface):
        """Display faults

        Parameters
        ----------
        surfacename : :obj:`str`
            Name of surface
        horizonset : :obj:`str`
            Name of horizon set
        title : :obj:`str`, optional
            Figure title (if ``None`` use ``surfacename``)
        addwells : :obj:`bool`, optional
            Add wells
        lims : :obj:`tuple`, optional
            y-x (or il-xl) limits
        savefig : :obj:`bool`, optional
            Save figure (``True``) or not (``False``)
        kwargs_surface : :obj:`dict`, optional
            Additional input parameters to be provided to
            :func:`pysubsurface.objects.Surface.view`

        Returns
        -------
        fig : :obj:`plt.figure`
           Figure handle (``None`` if ``axs`` are passed by user)
        ax : :obj:`plt.axes`
           Axes handle

        """
        fig, ax = self.view_surface(surfacename, horizonset, title=title,
                                    addwells=addwells, lims=lims,
                                    savefig=savefig, alpha=0.2,
                                    **kwargs_surface)
        axiskm = kwargs_surface.get('axiskm', False)
        for fault in self.faults.keys():
            self.faults[fault]['data'].view(color=self.faults[fault]['color'],
                                            axiskm=axiskm, ax=ax)
        # savefig
        if savefig:
            figpath = _create_figpath(self.figdir,
                                      os.path.join('geophysics',
                                                   'faults'))
            fig.savefig(
                os.path.join(figpath, self._projectname_nospace + '_faults.png'),
                dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor(),
                transparent=True)

        return fig, ax

    def view_wells_in_map(self, wellnames, ax, color='k'):
        """Display wells and line connecting them

        Parameters
        ----------
        wellnames : :obj:`list`
            Name of wells to display
        ax : :obj:`plt.axes`
            Axes handle
        color : :obj:`str`, optional
            Color of line

        Returns
        -------
        ax : :obj:`plt.axes`
           Axes handle

        """
        # Display wells
        for wellname in wellnames:
            self.wells[wellname].trajectory.view_traj(ax=ax, color=color)

        # Display line connecting wells
        xcoords = np.zeros(len(wellnames))
        ycoords = np.zeros_like(xcoords)
        for iwell, wellname in enumerate(wellnames):
            xcoords[iwell], ycoords[iwell] = self.wells[wellname].xcoord, \
                                             self.wells[wellname].ycoord
        ax.plot(xcoords, ycoords, color=color, lw=2)
        return ax

    def view_well_seismic(self, wellname, horizonset, seismicname, which='il',
                          tzoom=None, xzoom=None, cmap=cmap_amplitudepkdsg_r,
                          cbar=True, clip=0.3, clim=[], horlw=2, hornames=True,
                          level=None, shift=None, reservoir=None,
                          figsize=(30, 15), title=None, savefig=False):
        """Display well in seismic section

        Parameters
        ----------
        wellname : :obj:`str`
            Name of well to display
        horizonset : :obj:`str`
            Name of horizonset to display in seismic section
        seismicname : :obj:`str`
            Name of seismic cube to display
        which : :obj:`str`, optional
            ``IL``: display inline section passing through well,
            ``XL``: display crossline section passing through well.
            Note that if well is not vertical an arbitrary path along the well
            trajectory will be chosen
        tzoom : :obj:`tuple`, optional
            Time/depth start and end values (or indeces) for visualization
            of time/depth axis
        xzoom : :obj:`tuple`, optional
            Lateral dimension start and end values (or indeces)
            for visualization of IL/XL axis
        cmap : :obj:`str`, optional
            Colormap of seismic
        cbar : :obj:`bool`, optional
            Show colorbar
        clip : :obj:`str`, optional
            Clip of seismic data
        clim : :obj:`float`, optional
             Colorbar limits (if ``None`` infer from data and
             apply ``clip`` to those)
        hornames : :obj:`bool`, optional
             Add names of horizons (``True``) or not (``False``)
        level : :obj:`str`
            Level of interval(s) for picks display
        shift : :obj:`float`, optional
            Vertical shift to apply to picks
            (if ``None`` no shift is applied)
        reservoir : :obj:`dict`, optional
            Dictionary containing indices of ``top`` and ``base`` reservoir
            as well as the name of ``GOC`` and ``WOC`` to color-fill overlaid
            to seismic (if ``None`` do not color fill)
        figsize : :obj:`tuple`, optional
            Size of figure
        title : :obj:`str`, optional
            Title of figure
        savefig : :obj:`bool`, optional
            Save figure (``True``) or not (``False``)

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle (``None`` if ``axs`` are passed by user)
        axs : :obj:`plt.axes`
            Axes handles

        """
        fig, ax = self.wells[wellname].view_in_seismicsection(
            self.seismics[seismicname]['data'],
            domain=self.seismics[seismicname]['domain'],
            which=which, level=level,
            onlyinside=True, extend=30, logcurveshift=shift,
            **dict(cmap=cmap,
                   cbar=cbar, clip=clip, clim=clim,
                   interp='sinc',
                   tzoom_index=False,
                   tzoom=tzoom,
                   #xlzoom=xzoom,
                   #ilzoom=xzoom,
                   display_wellname=True,
                   title=seismicname if title is None else title,
                   horizons=self.horizonsets[horizonset]['data'],
                   horcolors=self.horizonsets[horizonset]['colors'],
                   horlw=horlw,
                   hornames=hornames,
                   reservoir=reservoir,
                   figsize=figsize))
        # savefig
        if savefig and fig is not None:
            figpath = _create_figpath(self.figdir, os.path.join('geophysics'))
            fig.savefig(os.path.join(figpath, self._projectname_nospace + '_' +
                                     change_name_for_unix(wellname) +
                                     change_name_for_unix(seismicname) +
                                     '_'+which+'.png'),
                        dpi=300, bbox_inches='tight')
        return fig, ax

    def view_well_overview(self, wellname, horizonset, surfacename,
                           seismicname, tzoom_seismic=None,
                           cmap_surface=cmap_hordsg,
                           cmap_seismic=cmap_amplitudepkdsg_r,
                           clip_seismic=0.3, extend=70, level=2,
                           figsize=(30, 15), title=None, savefig=False):
        """Display well in map and in seismic section for general well overview

        Parameters
        ----------
        wellname : :obj:`str`
            Name of well to display
        horizonset : :obj:`str`
            Name of horizonset to display in seismic section
        surfacename : :obj:`str`
            Name of surface in ``horizonset`` to display
        seismicname : :obj:`str`
            Name of seismic cube to display
        tzoom : :obj:`tuple`, optional
            Time/depth start and end values (or indeces) for visualization
            of time/depth axis
        cmap_surface : :obj:`str`, optional
            Colormap of surface
        cmap_seismic : :obj:`str`, optional
            Colormap of seismic
        clip_seismic : :obj:`str`, optional
            Clip of seismic data
        extend : :obj:`int`, optional
            Number of ilines and crosslines to add at the end of well toe when
            visualizing a deviated well
        level : :obj:`str`
            Level of interval(s) for picks display
        figsize : :obj:`tuple`, optional
            Size of figure
        title : :obj:`str`, optional
            Title of figure
        savefig : :obj:`bool`, optional
            Save figure (``True``) or not (``False``)

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle (``None`` if ``axs`` are passed by user)
        axs : :obj:`plt.axes`
            Axes handles

        """
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title if title is not None else
                     'Overview for well {}'.format(wellname), y=0.99,
                     fontsize=25, fontweight='bold')
        ax0 = plt.subplot2grid((1, 2), (0, 0))
        ax1 = plt.subplot2grid((1, 2), (0, 1))

        try:
            isurface = self.horizonsets[horizonset]['names'].index(surfacename)
        except:
            raise ValueError('{} not available in {}'.format(surfacename,
                                                             horizonset))

        # display well on surface
        if isinstance(self.horizonsets[horizonset]['data'], Interpretation):
            horizonplot = self.horizonsets[horizonset]['data'].surfaces[isurface]
        elif isinstance(self.horizonsets[horizonset]['data'], Ensemble):
            firstname = self.horizonsets[horizonset]['data'].firstintname
            horizonplot = \
                self.horizonsets[horizonset]['data'].interpretations[firstname].surfaces[isurface]
        _, _ = \
            horizonplot.view(
                ax=ax0, which='yx',
                cmap=cmap_surface, originlower=True,
                cbar=True, chist=True, nhist=101, ncountour=2,
                scalebar=True, axiskm=True, figsize=(8, 9),
                title=surfacename,
                titlesize=12)
        self.wells[wellname].trajectory.view_traj(ax=ax0, color='k',
                                                  axiskm=True,
                                                  labels=False,
                                                  fontsize=12,
                                                  wellname=False,
                                                  checkwell=True)
        if self.wells[wellname].perforations is not None:
            color = 'k' if self.wells[wellname].purpose == \
                           'production' else 'w'
            self.wells[wellname].perforations.view_in_map(ax=ax0,
                                                          scaling=1000.,
                                                          color=color)
        if self.wells[wellname].completions is not None:
            self.wells[wellname].completions.view_in_map(ax=ax0,
                                                         scaling=1000.)
        # display well on seismic
        _, ax1 = self.wells[wellname].view_in_seismicsection(
            self.seismics[seismicname]['data'],
            'depth', level=level, ax=ax1,
            onlyinside=True, extend=extend,
            **dict(fig=fig,
                   cmap=cmap_seismic,
                   cbar=True, clip=clip_seismic,
                   interp='sinc',
                   tzoom_index=False,
                   tzoom=tzoom_seismic,
                   display_wellname=False,
                   title=seismicname,
                   horizons=self.horizonsets[horizonset]['data'],
                   horcolors=self.horizonsets[horizonset]['colors']))

        # savefig
        if savefig:
            figpath = _create_figpath(self.figdir, os.path.join('wellplanning',
                                                                change_name_for_unix(wellname)))
            fig.savefig(os.path.join(figpath, change_name_for_unix(wellname) +
                                     '_overview.png'),
                        dpi=300, bbox_inches='tight')
        return fig, (ax0, ax1)

    def view_seismic_polarity(self, wellname, seismicname, pickname,
                              ailog='LFP_AI', horizonsetname=None,
                              cmap='seismic', ylim=None, normal=False,
                              shift=0., savefig=True):
        """Display polarity convention of seismic data.

        The acoustic impedance of well ``wellname`` is display alongside with
        a section of ``seismicname`` passing through the well and a specific
        pick is highlighed to identify the polarity convention used in the
        seismic data.

        wellname : :obj:`str`
            Name of well to display
        surfacename : :obj:`str`
            Name of surface in ``horizonset`` to display
        seismicname : :obj:`str`
            Name of seismic cube to display
        ailog : :obj:`str`
            Name of pick to use for polarity convention identification
        cmap_surface : :obj:`str`, optional
            Colormap of surface
        ailog : :obj:`str`, optional
            Name of welllog containing acoustic impedance
        ylim : :obj:`tuple`, optional
            Limits in depth axis
        normal : :obj:`bool`, optional
            Normal (``True``) or Reverse (``False``) polarity to use for plot
        shift : :obj:`tuple`, optional
            Vertical hifts to apply to picks
        savefig : :obj:`bool`, optional
            Save figure (``True``) or not (``False``)

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle (``None`` if ``axs`` are passed by user)
        axs : :obj:`plt.axes`
            Axes handles

        """
        fig = plt.figure(figsize=(18, 10))
        ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((2, 4), (0, 1), rowspan=2, colspan=2)

        top = self.wells[wellname].picks.df[
            self.wells[wellname].picks.df['Name'] == pickname].iloc[0]['TVDSS (meters)']
        color = self.wells[wellname].picks.df[
            self.wells[wellname].picks.df['Name'] == pickname].iloc[0]['Color']
        if ylim is None:
            ylim = (top*0.8, top*1.2)

        if horizonsetname is not None:
            kwargs_seismic = \
                {'horizons': self.horizonsets[horizonsetname]['data'],
                 'horcolors': self.horizonsets[horizonsetname]['colors'],
                 'horlw': 3}
        else:
            kwargs_seismic = {}

        _, ax1 = self.wells[wellname].welllogs.visualize_logcurve(
            ailog, 'TVDSS', ax=ax1, ylim=ylim)
        ax1.axhline(top, color=color, lw=3, ls='--')
        _, ax2 = self.wells[wellname].view_in_seismicsection(
            self.seismics[seismicname]['data'], level=None,
            logcurveshift=shift,
            display_wellname=False, cmap=cmap,
            title='Seismic along {}'.format(wellname),
            ax=ax2, **kwargs_seismic)
        ax2.set_ylim(ylim[1], ylim[0])
        ax2.set_yticks([])
        ax2.set_ylabel('pick')
        ax2.axhline(top, color=color, lw=3, ls='--')

        # add polarity cartoon
        polax = fig.add_axes([0.57, 0.15, 0.11, 0.18], anchor='SW')
        _seismic_polarity(polax, cmap=cmap, normal=normal, lw=2,
                          fs=8)

        # savefig
        if savefig:
            figpath = _create_figpath(self.figdir, 'geophysics')
            fig.savefig(
                os.path.join(figpath, self._projectname_nospace + '_' +
                             seismicname + '_polarityconvention.png'),
                dpi=300, bbox_inches='tight')
        axs = (ax1, ax2)
        return fig, axs
