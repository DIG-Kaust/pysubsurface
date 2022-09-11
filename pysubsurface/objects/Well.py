import logging
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from math import isnan
from collections import OrderedDict
from scipy.interpolate import interp1d

from pysubsurface.utils.utils import findclosest, unique_pairs, change_name_for_unix
from pysubsurface.utils.stats import covariance, drawsamples
from pysubsurface.utils.units import *
from pysubsurface.objects.utils import _findclosest_well_seismicsections
from pysubsurface.objects.Intervals import Intervals
from pysubsurface.objects.Picks import Picks
from pysubsurface.objects.Trajectory import Trajectory
from pysubsurface.objects.TDcurve import TDcurve
from pysubsurface.objects.Logs import _filters_curves, _visualize_facies, Logs
from pysubsurface.proc.seismicmod.poststack import zerooffset_wellmod, timeshift
from pysubsurface.proc.rockphysics.solid import Matrix, Rock
from pysubsurface.proc.rockphysics.fluid import Fluid
from pysubsurface.proc.rockphysics.gassmann import Gassmann
from pysubsurface.visual.utils import _wiggletrace, _wiggletracecomb


try:
    from IPython.display import display
    ipython_flag = True
except:
    ipython_flag=False


_vertical_thresh = 200 # allow 200 meters lateral offset between well head and well toe


class Well:
    """Well object.

    This object contains general information, trajectory, TD curves
    (and or checkshots), logs, picks and intervals for a single well.

    Parameters
    ----------
    projectdir : :obj:`str`
        Project data directory (where input files are located following
        the folder structure convention of pysubsurface)
    wellname : :obj:`str`
        Name of well to import (the same name used for all relevant input files
        containing trajectory and logs)
    rename : :obj:`str`, optional
        Name of well to import (overwritten and used to match picks)
    purpose : :obj:`str`, optional
            Purpose of well to filter on (can be wildcat, production,
            observation, injection or ``None``)
    gas_to_res : :obj:`float`, optional
        Factor to convert gas to reservoir condition (if ``None`` keep in
        surface conditions)
    readlogs : :obj:`bool`, optional
        Read welllogs from file (``True``) or not (``False``)
    kind : :obj:`str`, optional
        ``local`` when data are stored locally in a folder

    Attributes
    ----------
    xcoord : :obj:`float`
        UTM X coordinate of well
    ycoord : :obj:`float`
        UTM Y coordinate of well
    trajectory : :obj:`pysubsurface.objects.Trajectory`
        Well trajectory
    welllogs : :obj:`pysubsurface.objects.Logs`
        Well logs
    tdcurves : :obj:`dict`
        TD curves
    checkshots : :obj:`dict`
        Checkshots
    production : :obj:`dict`
        Production/Injection profiles

    """
    def __init__(self, projectdir, wellname, rename=None, purpose=None,
                 gas_to_res=None, readlogs=True, kind='local'):
        self.projectdir = projectdir
        self.purpose = purpose
        self.gas_to_res = gas_to_res
        self._readlogs = readlogs
        self._kind = kind

        if not self._kind in ['local', ]:
            NotImplementedError('kind must be local')

        # read well logs (check if file is available and raise a warning if not present...)
        wellname_logs = wellname

        if readlogs:
            logexists = \
                os.path.isfile(os.path.join(projectdir, 'Well', 'Logs',
                                            wellname_logs + '.las'))
            if logexists:
                self.welllogs = \
                    Logs(os.path.join(projectdir, 'Well/Logs/',
                                      wellname_logs + '.las'),
                         kind=kind)
            else:
                self.welllogs = None
                logging.warning('Logs file not found for {} '
                                'well...'.format(wellname))
        else:
            self.welllogs = None

        if rename is None:
            self.wellname = wellname
        else:
            self.wellname = rename

        # read well trajectory
        self.trajectory = \
            Trajectory(os.path.join(projectdir, 'Well/Trajectories/',
                                    wellname+'.csv'),
                       wellname=self.wellname, kind=self._kind)
        if self.trajectory.df is not None:
            self.xcoord = self.trajectory.df.iloc[0]['X Absolute']
            self.ycoord = self.trajectory.df.iloc[0]['Y Absolute']
            self.rkb = -self.trajectory.df.iloc[0]['TVDSS']

            # understand if well is vertical or not
            self._headtoedistance = \
                np.sqrt(np.sum((np.array([self.trajectory.df.iloc[0]['X Offset (meters)'],
                                          self.trajectory.df.iloc[0]['Y Offset (meters)']]) - \
                                np.array([self.trajectory.df.iloc[-1]['X Offset (meters)'],
                                          self.trajectory.df.iloc[-1]['Y Offset (meters)']]))**2))
            if self._headtoedistance < _vertical_thresh:
                self.vertical = True
            else:
                self.vertical = False
            # initialize checkshot and tdcurve containers
            self.tdcurves = {}
            self.checkshots = {}

        # initialize contacts
        self.contacts = None

        # initialize perforations and completions and tdcurve containers
        self.perforations = None
        self.completions = None

    def __str__(self):
        descr = 'Well {})\n\n'.format(self.wellname) + \
                'X: {}, Y:{}\n'.format(self.xcoord, self.ycoord) + \
                'TD curves: {}\n'.format(list(self.tdcurves.keys())) + \
                'Checkshots: {}\n'.format(list(self.checkshots.keys()))
        if self._readlogs:
            descr += 'Logs: {}\n'.format(list(self.welllogs.logs.keys()))
        return descr

    def add_picks(self, picks, computetvdss=False, step_md=0.1):
        """Add picks to well object

        Parameters
        ----------
        picks : :obj:`pysubsurface.objects.Picks`
            Object containing picks (for single well or entire field)
        computetvdss : :obj:`bool`, optional
            Compute TDVSS from picks and checkshots/TDcurve (``True``) or not
            Compute TDVSS from picks and checkshots/TDcurve (``True``) or not
            (``False``)
        step_md : :obj:`float`, optional
            Step size of interpolating MD curve

        """
        self.picks = picks.extract_from_well(self.wellname)
        if len(self.picks.df) > 0 and computetvdss:
            self.compute_picks_tvdss(step_md=step_md)

    def add_tdcurve(self, filename, name=None, checkshot=True):
        """Add TD curve (or checkshots) to well

        Parameters
        ----------
        filename : :obj:`str`
             Name of file containing TD curve or checkshots to be read
             (without extension)
        filename : :obj:`str`
             Name to give to TD curve or checkshots (if ``None`` use filename)
        checkshot : :obj:`bool`, optional
             Curve is CS (``True``) or not (``False``)

        """
        if checkshot:
            self.checkshots[name] = \
                TDcurve(os.path.join(self.projectdir, 'Well',
                                     'Checkshots', filename+'.csv'),
                        name=name, kind=self._kind)
        else:
            self.tdcurves[name] = \
                TDcurve(os.path.join(self.projectdir, 'Well', 'TDCurve',
                                     filename+'.csv'),
                        name=name, kind=self._kind)

    def add_production_injection(self):
        """Add production (or injection) profiles to well
        """
        self.production = Production(self._pdm, self.wellname,
                                     wellpurpose=self.purpose,
                                     gas_to_res=self.gas_to_res,
                                     kind=self._kind)

    def add_perforations(self):
        """Add perforations to well
        """
        self.perforations = Perforation(self._pdm, self.wellname,
                                        welltraj=self.trajectory,
                                        kind=self._kind)

    def add_completions(self):
        """Add completions to well
        """
        self.completions = Completion(self._pdm, self.wellname,
                                      welltraj=self.trajectory,
                                      kind=self._kind)


    def add_intervals_twt(self, twt_curve):
        """Add Two-Way Traveltime (TWT) to intervals dataframe of Well object
        given name of a currently available TWT curve in self.picks object

        Parameters
        ----------
        twt_curve : :obj:`str`
             Name of TWT curve in self.picks object

        """
        self.intervals.reset_index(drop=True, inplace=True)
        self.intervals['Top '+twt_curve] = None
        self.intervals['Base ' + twt_curve] = None

        for iinterval, interval in self.intervals.iterrows():
            pick_top_twt = self.picks.df[(self.picks.df['Name'] == interval['Top']) &
                                         (self.picks.df['Obs no'] == interval['Top obs'])][twt_curve].values
            pick_base_twt = self.picks.df[(self.picks.df['Name'] == interval['Base']) &
                                         (self.picks.df['Obs no'] == interval['Base obs'])][twt_curve].values
            # extract first value if multiple picks are repeated
            if len(pick_top_twt)>1:
                pick_top_twt = pick_top_twt[0]
                #print('Two top picks with same obs no')
                #print(self.picks.df[self.picks.df['Name'] == interval['Top']])
                #print(pick_top_twt)
            if len(pick_base_twt) > 1:
                pick_base_twt = pick_base_twt[0]
                #print('Two base picks with same obs no')
                #print(self.picks.df[self.picks.df['Name'] == interval['Base']])
                #print(pick_base_twt)
            self.intervals.loc[iinterval, 'Top '+twt_curve] = pick_top_twt
            self.intervals.loc[iinterval, 'Base '+twt_curve] = pick_base_twt

    def create_intervals(self, intervals):
        """Create intervals dataframe for specific well given an
        Intervals object

        Parameters
        ----------
        intervals : :obj:`pysubsurface.objects.Intervals`
            Intervals object

        """
        intervals_field = intervals.df[intervals.df['Field'] == self.picks.df.iloc[0]['Field']]
        if 'TVDSS (meters)' not in self.picks.df.columns:
            raise AttributeError('TVDSS (meters) not available in '
                                 'columns of picks.df, apply '
                                 'compute_picks_tvdss beforehand...')
        self.intervals = \
            pd.DataFrame(columns=['Name', 'Top', 'Base',
                                  'Top TVDSS (meters)', 'Base TVDSS (meters)',
                                  'Level', 'Color'])
        for iinterval, interval in intervals_field.iterrows():
            picks_top_md = \
                self.picks.df[self.picks.df['Name'] ==
                              interval['Top']]['Depth (meters)'].values
            picks_base_md = \
                self.picks.df[self.picks.df['Name'] ==
                              interval['Base']]['Depth (meters)'].values
            picks_top = \
                self.picks.df[self.picks.df['Name'] ==
                              interval['Top']]['TVDSS (meters)'].values
            picks_base = \
                self.picks.df[self.picks.df['Name'] ==
                              interval['Base']]['TVDSS (meters)'].values
            picks_top_obs = \
                self.picks.df[self.picks.df['Name'] ==
                              interval['Top']]['Obs no'].values
            picks_base_obs = \
                self.picks.df[self.picks.df['Name'] ==
                              interval['Base']]['Obs no'].values

            if len(picks_top) >= 1 and len(picks_base) >= 1:
                # when multiple picks are available use only last observation (largest Obs no)
                if len(picks_top) > 1:
                    pick_top_maxobs = np.where(picks_top_obs == np.max(picks_top_obs))[0][0]
                    picks_top_md = picks_top_md[pick_top_maxobs]
                    picks_top = picks_top[pick_top_maxobs]
                if len(picks_base) > 1:
                    pick_base_maxobs = np.where(picks_base_obs == np.max(picks_base_obs))[0][0]
                    picks_base_md = picks_base_md[pick_base_maxobs]
                    picks_base = picks_base[pick_base_maxobs]

                self.intervals = \
                    self.intervals.append(
                        pd.DataFrame({'Name': interval['Name'],
                                      'Top': interval['Top'],
                                      'Base': interval['Base'],
                                      'Top obs': np.max(picks_top_obs),
                                      'Base obs': np.max(picks_base_obs),
                                      'Top MD (meters)': picks_top_md,
                                      'Base MD (meters)': picks_base_md,
                                      'Top TVDSS (meters)': picks_top,
                                      'Base TVDSS (meters)': picks_base,
                                      'Level': interval['Level'],
                                      'Color': interval['Color']},
                                     index=[0]), ignore_index=True)
            else:
                if len(picks_top) == 0:
                    errormsg = 'cannot add {} interval: ' \
                               'no {} top pick...'.format(interval['Name'],
                                                          interval['Top'])
                elif len(picks_base) == 0:
                    errormsg = 'cannot add {} interval: ' \
                               'no {} base pick...'.format(interval['Name'],
                                                           interval['Base'])
                logging.warning(errormsg)
        self.intervals['Thickness (meters)'] = \
            self.intervals['Base TVDSS (meters)'] - \
            self.intervals['Top TVDSS (meters)']

        # check if exists any interval with negative thickness
        if np.sum(self.intervals['Thickness (meters)'] < 0) > 0:
            logging.warning('found Thickness (meters) smaller than zero')
        # remove intervals with negative thickness
        self.intervals = \
            self.intervals[self.intervals['Thickness (meters)'] >= 0]

    def create_contacts(self):
        """Create contacts object to fill with fluid contacts

        """
        self.contacts = Picks()

    def compute_picks_tvdss(self, step_md=0.1):
        """Compute true-vertical depth from sea surface (TDVSS)
        for each pick given checkshots or TDcurve as mapping function for
        MD-TVDSS

        Parameters
        ----------
        step_md : :obj:`float`, optional
            Step size of interpolating MD curve

        """
        self.picks.df.reset_index(inplace=True, drop=True)

        # create regular tvdss axis for mapping of picks
        self.picks.df.loc[:, 'TVDSS (meters)'] = np.nan
        md = self.trajectory.df['MD (meters)']
        tvdss = self.trajectory.df['TVDSS']

        md_in = md.min()
        md_end = md.max()
        md_reg = np.arange(md_in, md_end, step_md)

        f = interp1d(md,tvdss, kind='linear')
        tvdss_reg = f(md_reg)

        # mapping of picks from md to tvdss
        for index, row in self.picks.df.iterrows():
            imd = findclosest(md_reg, row['Depth (meters)'], checkoutside=False)
            if imd is not None:
                self.picks.df.loc[index, 'TVDSS (meters)'] = tvdss_reg[imd]

    def compute_picks_twt(self, step_md=0.1,
                          tdcurve_name=None,
                          checkshot_name=None, contacts=False):
        """Compute Two-Way Traveltime (TWT) for each pick given checkshots
        or TDcurve as mapping function for MD-TWT

        Parameters
        ----------
        step_md : :obj:`float`, optional
            Step size of interpolating MD curve
        tdcurve_name : :obj:`float`, optional
            Name of TD curve to use (if ``None`` use checkshot curve)
        checkshot_name : :obj:`float`, optional
            Name of checkshot curve to use (if ``None`` use TD curve)
        checkshot_name : :obj:`bool`, optional
            Compute twt for contacts (``True``) or picks (``False``)
        """
        tdcheck_name = tdcurve_name if tdcurve_name is not None \
            else checkshot_name
        if contacts:
            picks = self.contacts.df
        else:
            picks = self.picks.df

        picks.reset_index(inplace=True, drop=True)

        # create regular twt axis for mapping of picks
        picks.loc[:, 'TWT - {} (ms)'.format(tdcheck_name)] = np.nan

        if tdcurve_name is not None:
            md = self.tdcurves[tdcurve_name].df['Md (meters)']
            twt = self.tdcurves[tdcurve_name].df['Time (ms)']
        else:
            md = self.checkshots[checkshot_name].df['Md (meters)']
            twt = self.checkshots[checkshot_name].df['Time (ms)']

        # find picks TWT
        md_in = md.min()
        md_end = md.max()
        md_reg = np.arange(md_in, md_end, step_md)

        f = interp1d(md, twt, kind='linear')
        twt_reg = f(md_reg)

        # mapping of picks from md to TWT
        for index, row in picks.iterrows():
            if not np.isnan(row['Depth (meters)']):
                imd = findclosest(md_reg, row['Depth (meters)'], checkoutside=True)
                if imd is not None:
                    picks.loc[index, 'TWT - {} (ms)'.format(tdcheck_name)] = twt_reg[imd]
        # save back into object
        if contacts:
            self.contacts.df = picks
        else:
            self.picks.df = picks

    def compute_logs_tvdss(self):
        """Add TVDSS curve to wellogs object
        using MD in welllogs and MD+TVDSS in trajectory
        """
        self.welllogs.add_tvdss(self.trajectory)

    def compute_logs_twt(self, tdcurve_name=None, checkshot_name=None):
        """Add TWT curve to wellogs object using MD in welllogs
        and MD+TWT in trajectory

        Parameters
        ----------
        tdcurve_name : :obj:`float`, optional
            Name of TD curve to use (if ``None`` use checkshot curve)
        checkshot_name : :obj:`float`, optional
            Name of checkshot curve to use (if ``None`` use TD curve)

        """
        if tdcurve_name is not None:
            self.welllogs.add_twt(self.tdcurves[tdcurve_name], tdcurve_name)
        else:
            self.welllogs.add_twt(self.checkshots[checkshot_name], checkshot_name)

    def assign_facies(self, faciesset):
        """Assign facies mapping to all wells and write a new welllog with
        facies definition.

        Parameters
        ----------
        faciesset : :obj:`dict`
            Facies set, dictionary of :class:`pysubsurface.objects.Facies`
            objects

        """
        facieslog = np.full(self.welllogs.df.shape[0], np.nan)

        faciesnames = faciesset.keys()
        for ifacies, faciesname in enumerate(faciesnames):
            facies = faciesset[faciesname]
            mask = facies.extract_mask_from_well(self)
            facieslog[mask] = ifacies
        self.welllogs.delete_curve('Facies')
        self.welllogs.add_curve(facieslog, 'Facies', descr='Facies')
        self.welllogs.dataframe()

    def return_custom_intervals(self, intervals, tops, colors=None):
        """Create custom intervals based on set of tops and bases.

        This routines can be used to aggregate different intervals in the
        stratigraphic column and return a custom set of intervals

        Parameters
        ----------
        intervals : :obj:`list`
            List of interval names
        tops : :obj:`list`
            List of tops to be used to define custom intervals
            (top_0 - top_1, top_1 - top_2..., top_n-1 - top_n)
        colors : :obj:`list`
            List of colors for custom intervals

        """
        df = pd.DataFrame(columns=(['Name', 'Base', 'Top', 'Base MD (meters)',
                                    'Base TVDSS (meters)', 'Top MD (meters)',
                                    'Top TVDSS (meters)', 'Thickness (meters)',
                                    'Top TWT (ms)', 'Base TWT (ms)',
                                    'TWT Thickness (ms)', 'Velocity (m/s)',
                                    'Color']))
        df = df.append([0]*len(intervals))
        df = df.drop(0, axis=1)

        if colors is None:
            colors = ['k'] * len(intervals)

        bases, tops = tops[1:], tops[:-1]
        for iint, (interval, top, base, color) in \
                enumerate(zip(intervals, tops, bases, colors)):
            df['Name'].iloc[iint] = interval
            df['Color'].iloc[iint] = color

            base_row = self.picks.df[self.picks.df['Name'] == base]
            top_row = self.picks.df[self.picks.df['Name'] == top]

            if len(base_row) > 0:
                base_row = base_row.iloc[0]
                df['Base'].iloc[iint] = base
                df['Base MD (meters)'].iloc[iint] = base_row['Depth (meters)']
                df['Base TVDSS (meters)'].iloc[iint] = base_row['TVDSS (meters)']
            if len(top_row) > 0:
                top_row = top_row.iloc[0]
                df['Top'].iloc[iint] = top
                df['Top MD (meters)'].iloc[iint] = top_row['Depth (meters)']
                df['Top TVDSS (meters)'].iloc[iint]= top_row['TVDSS (meters)']
            if len(base_row) > 0 and len(top_row) > 0:
                df['Thickness (meters)'].iloc[iint] = \
                    df['Base TVDSS (meters)'].iloc[iint] - df['Top TVDSS (meters)'].iloc[iint]
                # Add TWT Thickness if time is present
                top_twt = list(filter(lambda x: 'Top TWT' in x, top_row.index))
                base_twt = list(filter(lambda x: 'Top TWT' in x, base_row.index))
                if len(top_twt) == 1 and len(base_twt) == 1:
                    df['Top TWT (ms)'].iloc[iint] = top_row[top_twt].values[0]
                    df['Base TWT (ms)'].iloc[iint] = base_row[base_twt].values[0]
                    df['TWT Thickness (ms)'].iloc[iint] = \
                        df['Base TWT (ms)'].iloc[iint] - \
                        df['Top TWT (ms)'].iloc[iint]
                    df['Velocity (m/s)'].iloc[iint] = \
                        df['Thickness (meters)'].iloc[iint] / (df['TWT Thickness (ms)'].iloc[iint] / 2000.)
        return df

    def extrac_prop_in_interval(self, interval, level, property,
                                tops=None, addthickness=False):
        """Extract property and its x-y coordinates from ``intervals`` dataframe
        in the middle of a specified interval

        Parameters
        ----------
        interval : :obj:`str`
            Name of interval
        level : :obj:`int`, optional
            Level to analyze
        property : :obj:`str`, optional
            Name of property to extract from self.intervals dataframe
        tops : :obj:`list`
            Top and base picks for custom interval (i.e., not available in stratigraphic column)
        addthickness : :obj:`bool`, optional
            Find properties in the middle of an interval by adding thickness
            (``True``) or  not (``False``)

        Returns
        -------
        xcoord : :obj:`float`
            X coordinate (``None`` if chosen interval is not available
            in self.intervals)
        ycoord : :obj:`float`
            Y coordinate (``None`` if chosen interval is not available
            in self.intervals)
        prop : :obj:`float`
            Property (``None`` if chosen interval is not available
            in self.intervals)

        """
        if tops is None:
            wellint = self.intervals[self.intervals['Name'] == interval]
            wellint = wellint[wellint['Level'] == level]
        else:
            wellint = self.return_custom_intervals(interval, tops)
        thickness = wellint['Thickness (meters)'].values
        if property in wellint.columns:
            prop = wellint[property].values
        else:
            prop = []

        # only if one present assing to the list and find x-y locations
        if len(prop) == 1 and not isnan(prop) and not isnan(thickness):
            prop = float(prop)
            top = float(wellint['Top MD (meters)'])
            if addthickness:
                top += float(thickness) / 2
            iclosest = findclosest(self.trajectory.df['MD (meters)'], top)
            xcoord = \
                self.trajectory.df.iloc[iclosest]['X Absolute']
            ycoord = \
                self.trajectory.df.iloc[iclosest]['Y Absolute']
        else:
            prop = xcoord = ycoord = None

        return xcoord, ycoord, prop

    def extract_logs_in_interval(self, interval, lognames, filters=None):
        """Extract log(s) within a specified interval

        Parameters
        ----------
        interval : :obj:`pd.Series` or :obj:`pd.DataFrame`
            Interval to use for extraction
        lognames : :obj:`str` or :obj:`tuple`, optional
            Name(s) of log curve(s) to be extracted within inteval
        filters : :obj:`tuple`, optional
            Filters to be applied during extraction
            (each filter is a dictionary with logname, rule and chain, e.g.
            `logname='LFP_COAL'`, `rule='<0.1'`, `chain='and'/'or'`
            will keep all values where values in LFP_COAL logs are <0.1 and
            will be combined with additional rules with an and/or conditional)
            Note that chaining rule is strictly not needed for the first filter

        Returns
        -------
        log_in_int : :obj:`pd.Series` or :obj:`pd.DataFrame`
            Logs in chosen interval

        """
        interval = pd.Series(interval)

        # identify interval in log axis
        logs = self.welllogs.df
        logs_md = logs.index

        top_md = interval['Top MD (meters)']
        base_md = interval['Base MD (meters)']

        itop = findclosest(logs_md.values, top_md)
        ibase = findclosest(logs_md.values, base_md)

        # handle case where both markers are above start
        if itop == 0 and ibase == 0:
            log_in_int = []
        # handle case where both markers are below start
        elif itop == (len(logs_md) -1) and ibase == (len(logs_md) -1):
            log_in_int = []
        else:
            # apply filters
            if filters is not None:
                log_in_int = _filters_curves(logs.iloc[itop:ibase+1],
                                             filters)[0][lognames]
            else:
                log_in_int = logs.iloc[itop:ibase+1][lognames]
        return log_in_int

    def create_averageprops_intervals(self, level=2, intervals=None,
                                      vpname='LFP_VP',
                                      vsname='LFP_VS',
                                      rhoname='LFP_RHOB', ainame=None,
                                      vpvsname=None, filters=None):
        """Compute statistics for elastic properties
        within intervals (and save in self.welllogs object)

        Parameters
        ----------
        level : :obj:`int`, optional
            Level to analyze
        intervals : :obj:`pandas.DataFrame`, optional
            Custom intervals created via
            :func:`pysubsurface.objects.Well.return_custom_intervals`
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

        Returns
        -------
        averaged_props : :obj:`dict` or :obj:`pd.DataFrame`
            Statistics for every interval in chosen level

        """
        if intervals is None:
            if level is None:
                intervals = self.intervals[self.intervals['Level'].isnull()]
            else:
                if not isinstance(level, (list, tuple)):
                    level = [level]
                intervals = self.intervals[self.intervals['Level'].isin(level)]

        self.averaged_props = {vpname: {}, vsname: {}, rhoname: {},
                               'Cov': {}, 'Nsamples': {}}
        if ainame:
            self.averaged_props[ainame] = {}
        if vpvsname:
            self.averaged_props[vpvsname] = {}

        logs = self.welllogs.df
        logs_md = logs.index

        logs[vpname + '_mean'] = np.nan
        logs[vsname + '_mean'] = np.nan
        logs[rhoname + '_mean'] = np.nan

        for index, interval in intervals.iterrows():
            top_md = interval['Top MD (meters)']
            base_md = interval['Base MD (meters)']
            if top_md is not np.nan and base_md is not np.nan:
                # identify interval in log axis
                itop = findclosest(logs_md.values, top_md)
                ibase = findclosest(logs_md.values, base_md)

                if filters is None:
                    vpselect = logs.iloc[itop:ibase][vpname]
                    vsselect = logs.iloc[itop:ibase][vsname]
                    rhoselect = logs.iloc[itop:ibase][rhoname]
                else:
                    vpselect = _filters_curves(logs.iloc[itop:ibase],
                                               filters=filters)[0][vpname]
                    vsselect = _filters_curves(logs.iloc[itop:ibase],
                                               filters=filters)[0][vsname]
                    rhoselect = _filters_curves(logs.iloc[itop:ibase],
                                                filters=filters)[0][rhoname]
            else:
                vpselect = vsselect = rhoselect = np.array([])

            self.averaged_props[vpname][interval['Name']] = \
                {'mean': np.nanmean(vpselect), 'stdev': np.nanstd(vpselect)}
            self.averaged_props[vsname][interval['Name']] = \
                {'mean': np.nanmean(vsselect), 'stdev': np.nanstd(vsselect)}
            self.averaged_props[rhoname][interval['Name']] = \
                {'mean': np.nanmean(rhoselect), 'stdev': np.nanstd(rhoselect)}
            if ainame:
                self.averaged_props[ainame][interval['Name']] = \
                    {'mean': np.nanmean(rhoselect * vpselect),
                     'stdev': np.nanstd(rhoselect * vpselect)}
            if vpvsname:
                self.averaged_props[vpvsname][interval['Name']] = \
                    {'mean': np.nanmean(vpselect / vsselect),
                     'stdev': np.nanstd(vpselect / vsselect)}
            self.averaged_props['Cov'][interval['Name']] = \
                covariance([vpselect, vsselect, rhoselect],
                           featurenames=[vpname, vsname, rhoname])
            self.averaged_props['Nsamples'][interval['Name']] = \
                len(~np.isnan(vpselect))

            if top_md is not np.nan and base_md is not np.nan:
                logs[vpname + '_mean'].iloc[itop:ibase] = \
                    self.averaged_props[vpname][interval['Name']]['mean']
                logs[vsname + '_mean'].iloc[itop:ibase] = \
                    self.averaged_props[vsname][interval['Name']]['mean']
                logs[rhoname + '_mean'].iloc[itop:ibase] = \
                    self.averaged_props[rhoname][interval['Name']]['mean']
        self.averaged_props_summary = []
        for key in self.averaged_props.keys():
            if key is not 'Cov' and key is not 'Nsamples':
                dftmp = pd.DataFrame(self.averaged_props[key]).T
                dftmp = dftmp.add_prefix(key + '_')
                self.averaged_props_summary.append(dftmp)
        self.averaged_props_summary = pd.concat(self.averaged_props_summary,
                                                axis=1)

        # add curves to welllogs
        self.welllogs.df = logs
        self.welllogs.delete_curve(vpname + '_mean')
        self.welllogs.delete_curve(vsname + '_mean')
        self.welllogs.delete_curve(rhoname + '_mean')
        self.welllogs.add_curve(logs[vpname + '_mean'], vpname + '_mean',
                                descr='VP averaged in intervals')
        self.welllogs.add_curve(logs[vsname + '_mean'], vsname + '_mean',
                                descr='VS averaged in intervals')
        self.welllogs.add_curve(logs[rhoname + '_mean'], rhoname + '_mean',
                                descr='RHOB averaged in intervals')
        if ainame is not None:
            self.welllogs.delete_curve(ainame + '_mean')
            self.welllogs.add_curve(
                logs[vpname + '_mean'] * logs[rhoname + '_mean'],
                ainame + '_mean',
                descr='AI averaged in intervals')
        if vpvsname is not None:
            self.welllogs.delete_curve(vpvsname + '_mean')
            self.welllogs.add_curve(
                logs[vpname + '_mean'] / logs[vsname + '_mean'],
                vpvsname + '_mean',
                descr='VPVS averaged in intervals')
        return self.averaged_props

    def fluid_substitution(self, sand, shale, oil, water, changes, coal=None,
                           carb=None, gas=None, porocutoff=[0, 1.],
                           vshcutoff=[0., 1.], lfp=False,
                           phi='PHIT', vsh='VSH', vcoal='VCOAL', vcarb='VCARB',
                           vp='VP', vs='VS', rho='RHOB', ai='AI', vpvs='VPVS',
                           sot='SOT', sgt='SGT', timeshiftpp=True,
                           savelogs=True, savedeltas=True,
                           logssuffix='fluidsub'):
        """Gassmann fluid substitution on well logs.

        Parameters
        ----------
        sand : :obj:`dict`
            Bulk modulus and density of sand in dictionary ``{'k': X, 'rho': X}``
        shale : :obj:`dict`
            Bulk modulus and density of shale in dictionary ``{'k': X, 'rho': X}``
        oil : :obj:`pysubsurface.proc.rockphysics.fluid.Oil`
            Oil object
        water : :obj:`pysubsurface.proc.rockphysics.fluid.Brine`
            Brine object
        change : :obj:`dict` or :obj:`list`
            Changes to be applied to saturation logs in dictionary(ies)
            ``{'zmin': X or pickname, 'zmax': X or pickname, 'sot': X, 'sgt': X}``
            where
        coal : :obj:`dict`, optional
            Bulk modulus and density of coal in dictionary ``{'k': X, 'rho': X}``
        carb : :obj:`dict`, optional
            Bulk modulus and density of carbonate/calcite in dictionary
            ``{'k': X, 'rho': X}``
        gas : :obj:`pysubsurface.proc.rockphysics.fluid.Gas`, optional
            Gas object
        lfp : :obj:`bool`, optional
            Prepend `LFP_`` to every log (``True``) or not (``False``)
        phi : :obj:`str`, optional
            Name of Porosity log
        vsh : :obj:`str`, optional
            Name of gamma ray log
        vcoal : :obj:`str`, optional
            Name of Volume Coal log
        vcarb : :obj:`str`, optional
            Name of Volume Carbonate log
        vp : :obj:`str`, optional
            Name of P-wave velocity log
        vs : :obj:`str`, optional
            Name of S-wave velocity log
        vs : :obj:`str`, optional
            Name of S-wave velocity log
        rho : :obj:`str`, optional
            Name of Density log
        ai : :obj:`str`, optional
            Name of Acoustic Impedence log
        vpvs : :obj:`str`, optional
            Name of VP/VS log
        sot : :obj:`str`, optional
            Name of Total Oil Saturation log
        sgt : :obj:`str`, optional
            Name of Total Gas Saturation Ray log
        timeshiftpp : :obj:`bool`, optional
            Compute PP timeshift
        savelogs : :obj:`bool`, optional
            Save fluid substituted profiles as logs
        savedeltas : :obj:`bool`, optional
            Save differences as logs
        logssuffix : :obj:`str`, optional
            Suffix to add to log names if saved

        Returns
        -------
        vp1 : :obj:`numpy.ndarray`
            Fluid-substituted P-wave velocity
        vs1 : :obj:`numpy.ndarray`
            Fluid-substituted S-wave velocity
        rho1 : :obj:`numpy.ndarray`
            Fluid-substituted density
        so1 : :obj:`numpy.ndarray`
            Fluid-substituted oil saturation
        sg1 : :obj:`numpy.ndarray`
            Fluid-substituted gas saturation

        """
        # prepend lfp if lfp flag is True
        if lfp:
            vsh = 'LFP_' + vsh if lfp else vsh
            vcarb = 'LFP_' + vcarb if lfp else vcarb
            vcoal = 'LFP_' + vcoal if lfp else vcoal
            sgt = 'LFP_' + sgt if lfp else sgt
            sot = 'LFP_' + sot if lfp else sot
            phi = 'LFP_' + phi if lfp else phi
            vp = 'LFP_' + vp if lfp else vp
            vs = 'LFP_' + vs if lfp else vs
            rho = 'LFP_' + rho if lfp else rho
            ai = 'LFP_' + ai if lfp else ai
            vpvs = 'LFP_' + vpvs if lfp else vpvs
        vpname, vsname, rhoname, ainame, vpvsname, sotname, sgtname, = \
            vp, vs, rho, ai, vpvs, sot, sgt

        # extract logs
        z = self.welllogs.logs.index
        phi = self.welllogs.logs[phi].copy()
        vp = self.welllogs.logs[vp].copy()
        vs = self.welllogs.logs[vs].copy()
        rho = g_cm3_to_kg_m3(self.welllogs.logs[rho].copy())
        vsh = self.welllogs.logs[vsh].copy()
        vsand = 1. - vsh
        so0 = self.welllogs.logs[sot].copy()
        sw0 = 1. - so0
        if carb is not None:
            vcarb = self.welllogs.logs[vcarb]
            vsand -= vcarb
        else:
            vcarb = np.zeros_like(vsh)
        if coal is not None:
            vcoal = self.welllogs.logs[vcoal]
            vsand -= vcoal
        else:
            vcoal = np.zeros_like(vsh)
        if sgt is not None:
            sg0 = self.welllogs.logs[sgt]
            sw0 -= sg0
        else:
            sg0 = np.zeros_like(vsh)

        # cutoffs
        cutoff = np.zeros(len(z)).astype(bool)
        if porocutoff[0] > 0. or porocutoff[1] < 1.:
            cutoff = cutoff | (phi < porocutoff[0]) | (phi > porocutoff[1])
        if vshcutoff[0] > 0. or vshcutoff[1] < 1.:
            cutoff = cutoff | (vsh < vshcutoff[0]) | (vsh > vshcutoff[1])

        # fix nans fpr elastic params
        nans = cutoff | np.isnan(vp) | np.isnan(vs) | np.isnan(rho) | np.isnan(phi) | \
               np.isnan(vsh) | np.isnan(so0) | np.isnan(sg0)
        vp[nans] = 0
        vs[nans] = 0
        rho[nans] = 0
        phi[nans] = 0

        # apply changes to fluids
        sg1 = sg0.copy()
        so1 = so0.copy()
        if not isinstance(changes, list):
            changes = [changes]
        for change in changes:
            if isinstance(change['zmin'], str):
                change['zmin'] = \
                    self.picks.df[self.picks.df['Name'] == change['zmin']].iloc[0]['Depth (meters)']
            if isinstance(change['zmax'], str):
                change['zmax'] = \
                    self.picks.df[self.picks.df['Name'] == change['zmax']].iloc[0]['Depth (meters)']
            izmin = findclosest(z, change['zmin'])
            izmax = findclosest(z, change['zmax'])

            if 'sot' in change.keys():
                so1[izmin:izmax] = change['sot']
            if 'sgt' in change.keys():
                sg1[izmin:izmax] = change['sgt']
            sw1 = 1 - so1 - sg1
        so1_filled = so1.copy()
        sg1_filled = sg1.copy()
        sw1_filled = sw1.copy()
        so0[nans] = 0
        sg0[nans] = 0
        sw0[nans] = 0
        so1_filled[nans] = 0
        sg1_filled[nans] = 0
        sw1_filled[nans] = 0

        # create matrix and fluid
        sand['frac'] = vsand
        shale['frac'] = vsh
        minerals = {'sand': sand, 'shale': shale}

        if coal is not None:
            coal['frac'] = vcoal
            minerals['coal'] = coal
        if carb is not None:
            carb['frac'] = vcarb
            minerals['carb'] = carb
        mat = Matrix(minerals)

        if gas is None:
            fluid0 = Fluid({'oil': (oil, so0),
                            'water': (water, sw0)})
            fluid1 = Fluid({'oil': (oil, so1_filled),
                            'water': (water, sw1_filled)})
        else:
            fluid0 = Fluid({'gas': (gas, sg0),
                            'oil': (oil, so0),
                            'water': (water, sw0)})
            fluid1 = Fluid({'gas': (gas, sg1_filled),
                            'oil': (oil, so1_filled),
                            'water': (water, sw1_filled)})

        # fluid substitution
        medium0 = Rock(vp, vs, rho, mat, fluid0, poro=phi)
        fluidsub = Gassmann(medium0, fluid1, mask=True)

        # fill with original values in cutoff regions
        vp1, vs1, rho1 = fluidsub.medium1.vp, fluidsub.medium1.vs, fluidsub.medium1.rho
        vp1[cutoff] = self.welllogs.logs[vpname][cutoff]
        vs1[cutoff] = self.welllogs.logs[vsname][cutoff]
        rho1[cutoff] = g_cm3_to_kg_m3(self.welllogs.logs[rhoname])[cutoff]

        # save logs
        if savelogs:
            self.welllogs.add_curve(so1, '{}_{}'.format(sotname, logssuffix),
                                    unit='frac',
                                    descr='{} - {}'.format(sotname, logssuffix))
            self.welllogs.add_curve(sg1, '{}_{}'.format(sgtname, logssuffix),
                                    unit='frac',
                                    descr='{} - {}'.format(sotname, logssuffix))
            self.welllogs.add_curve(vp1, '{}_{}'.format(vpname, logssuffix),
                                    unit='m/s',
                                    descr='{} - {}'.format(vpname, logssuffix))
            self.welllogs.add_curve(vs1, '{}_{}'.format(vsname, logssuffix),
                                    unit='m/s',
                                    descr='{} - {}'.format(vsname, logssuffix))
            self.welllogs.add_curve(kg_m3_to_g_cm3(rho1),
                                    '{}_{}'.format(rhoname, logssuffix),
                                    unit='g/cm3',
                                    descr='{} - {}'.format(rhoname, logssuffix))
            self.welllogs.add_curve(vp1 * kg_m3_to_g_cm3(rho1),
                                    '{}_{}'.format(ainame, logssuffix),
                                    unit=None,
                                    descr='{} - {}'.format(ainame, logssuffix))
            self.welllogs.add_curve(vp1 / vs1, '{}_{}'.format(vpvsname, logssuffix),
                                    unit=None,
                                    descr='{} - {}'.format(vpvsname, logssuffix))
        if savedeltas:
            # differences
            vp = self.welllogs.logs[vpname].copy()
            vs = self.welllogs.logs[vsname].copy()
            rho = g_cm3_to_kg_m3(self.welllogs.logs[rhoname].copy())

            self.welllogs.add_curve(
                sg1 - self.welllogs.df[sgtname].values.copy(),
                '{}diff_{}'.format(sgtname, logssuffix),
                unit='frac',
                descr='d{} - {}'.format(sgtname, logssuffix))
            self.welllogs.add_curve(
                sg1 - self.welllogs.df[sgtname].values.copy(),
                '{}diff_{}'.format(sgtname, logssuffix),
                unit='frac',
                descr='d{} - {}'.format(sgtname, logssuffix))
            self.welllogs.add_curve(
                200 * (vp1 * kg_m3_to_g_cm3(rho1) - vp * kg_m3_to_g_cm3(rho)) / \
                (vp1 * kg_m3_to_g_cm3(rho1) + vp * kg_m3_to_g_cm3(rho)),
                '{}diff_{}'.format(ainame, logssuffix),
                unit='frac',
                descr='{} - {}'.format(ainame, logssuffix))
            self.welllogs.add_curve(200 * (vp1 / vs1  - vp / vs) / (vp1 / vs1  + vp / vs),
                                    '{}diff_{}'.format(vpvsname, logssuffix),
                                    unit = 'frac',
                                    descr = 'd{} - {}'.format(vpvsname,
                                                              logssuffix))
        # timeshifts
        if timeshiftpp:
            tpp = timeshift(z, vp, vp1)
            self.welllogs.add_curve(tpp, 'PPtimeshift_{}'.format(logssuffix),
                unit='ms',
                descr='PPtimeshift - {}'.format(logssuffix))

        return vp1, vs1, rho1, so1, sg1


    #########
    # Viewers
    #########
    def display(self, nrows=10):
        """Display various data belonging to well object

        Parameters
        ----------
        nrows : :obj:`int`, optional
            Number of rows to display (if ``None`` display
        """
        print('{:<15s}{:<20s}'.format('Wellname', self.wellname))
        print('{:<15s}x   = {:<14f} - y   = {:<14f}'.format('Location',
                                                            self.xcoord,
                                                            self.ycoord))
        print('')
        print('\033[1mTrajectory\033[0m')
        self.trajectory.display(nrows=nrows)
        print('')
        print('\033[1mPicks\033[0m')
        self.picks.display(nrows=nrows)
        if self.intervals is not None:
            print('')
            print('\033[1mIntervals\033[0m')
            if ipython_flag :
                display(self.intervals.head(nrows).style.
                        applymap(lambda x: 'background-color: {}'.format(x),
                                 subset=['Color']))
            else:
                print(self.intervals.head(nrows))
        else:
            print('\033[1mNo Intervals\033[0m')
        if hasattr(self, 'checkshots') and len(self.checkshots) > 0:
            print('\033[1mCheckshots\033[0m')
            for _, checkshot in self.checkshots.items():
                print('Name: %s' % checkshot.name)
                checkshot.display(nrows=nrows)
        else:
            print('\033[1mNo Checkshots\033[0m')
        if hasattr(self, 'tdcurves') and len(self.tdcurves) > 0:
            print('\033[1mTD Curves\033[0m')
            for _, tdcurve in self.tdcurves.items():
                print('Name: %s' % tdcurve.name)
                tdcurve.display(nrows=nrows)
        else:
            print('\033[1mNo TD Curves\033[0m')
        if hasattr(self, 'welllogs'):
            print('')
            print('\033[1mWelllogs\033[0m')
            self.welllogs.display(nrows=nrows)
        else:
            print('\033[1mNo Welllogs\033[0m')

        if hasattr(self, 'averaged_props'):
            print('\033[1mAveraged Properties\033[0m')
            if ipython_flag:
                display(self.averaged_props_summary)
            else:
                print(self.averaged_props_summary)

    def view(self, ax=None, level=None, labels=False, inverty=False):
        """Visualize well trajectory with completions and perforations

        Parameters
        ----------
        ax : :obj:`plt.axes`
           Axes handle
        level : :obj:`str`
            Level of interval(s)
        labels : :obj:`bool`, optional
            Display name of picks
        inverty : :obj:`bool`, optional
            Invert y axis

        Returns
        -------
        fig : :obj:`plt.figure`
           Figure handle (``None`` if ``axs`` are passed by user)
        ax : :obj:`plt.axes`
           Axes handle

        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 15))
        else:
            fig = None

        ntraj = len(self.trajectory.df)
        ax.plot(np.zeros(ntraj),#self.trajectory.df['X Absolute'],
                self.trajectory.df['MD (meters)'], 'k')

        for perf in self.perforations.perforations:
            ntraj = len(perf)
            ax.plot(np.zeros(ntraj), #perf['X Absolute'],
                    perf['MD (meters)'], 'k', lw=11,
                    label='Perforation')
            ax.plot(np.zeros(ntraj), #perf['X Absolute'],
                    perf['MD (meters)'], 'w', lw=10,
                    label='Perforation')
        if self.intervals is not None:
            _ = self.view_picks_and_intervals(ax, level=level, depth='MD',
                                              filtname='Base',
                                              showintervals=True,
                                              labels=labels)

        self.completions.df.reset_index(drop=True, inplace=True)
        for icomp, comp in self.completions.df.iterrows():
            ntraj = len(self.completions.completions[icomp])
            ax.plot(np.zeros(ntraj), #self.completions.completions[icomp]['X Absolute'],
                    self.completions.completions[icomp]['MD (meters)'],
                    _completions[comp['symbol_name']][0],
                    lw=_completions[comp['symbol_name']][1],
                    label=comp['symbol_name'])
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels[1:], handles[1:]))
        ax.legend(by_label.values(), by_label.keys())
        ax.set_xlabel('X Absolute [meters]')
        ax.set_ylabel('MD [meters]')
        ax.set_title(self.wellname)
        if inverty:
            ax.invert_yaxis()
        ax.set_xlim(-0.04, 0.04)
        return fig, ax

    def view_picks_and_intervals(self, axs, depth='Depth', filtname=None,
                                 twtcurve=None, ylim=None, level=2,
                                 showintervals=True,
                                 skiplast=False, labels=True, pad=1.02):
        """Visualize picks and intervals on top of log curves

        For this routine to work sensibly, axes handles
        ``axs`` should be previously obtained from the
        :func:`pysubsurface.objects.Logs.visualize_logcurve` or
        :func:`pysubsurface.objects.Logs.visualize_logcurves` routines.

        Parameters
        ----------
        axs : :obj:`plt.axes`
            Axes handles
        depth : :obj:`str`, optional
            Domain to use for plotting of picks
            (``Depth``: MD, ``TVDSS``: TVDSS, ``TWT``: TWT)
        filtname : :obj:`str` or  :obj:`list`, optional
            Regex filters to be applied to picks to be shown column
        twtcurve : :obj:`tuple`, optional
            Name of TWT curve to display when ``depth=TWT``
        ylim : :obj:`tuple`, optional
            Extremes of y-axis (picks outside those extremes will be discarded)
        level : :obj:`int` or :obj:`list`, optional
            Interval level(s) to display (or dictionary containing a list of
            custom interval names and interval tops to be passed to
            :func:`pysubsurface.objects.Well.return_custom_intervals` to create a
            custom Interval object.
        showintervals : :obj:`bool`, optional
            Display also intervals
        skiplast : :obj:`bool`, optional
            Do not show intervals in the last panel (``True``) or show in all
            panels (``False``)
        labels : :obj:`bool`, optional
            Display name of picks
        pad : :obj:`float`, optional
            Padding for labels

        Returns
        -------
        axs : :obj:`plt.axes`
            Axes handles

        """
        if not isinstance(axs, (list, np.ndarray)):
            axs = np.array([axs])

        # extract only picks of interest
        if not isinstance(level, (list, tuple, dict)):
            level = [level]

        if isinstance(level, dict):
            intervals_plot = \
                Intervals(self.return_custom_intervals(level['Name'],
                                                       level['Top'],
                                                       level['Color']))
        else:
            intervals_plot = \
                Intervals(self.intervals[self.intervals['Level'].isin(level)])
        #picksname_plot = \
        #    (intervals_plot.df['Top'].append(intervals_plot.df['Base'])).unique()
        picksname_plot = intervals_plot.df['Top'].unique()
        picks_plot = \
            Picks(picks=self.picks.df[self.picks.df['Name'].isin(picksname_plot)])

        # filter picks
        if filtname is not None:
            picks_plot.discard_on_keywords(filtname if isinstance(filtname, list)
                                           else[filtname])
        # identify labels for picks and intervals
        if depth == 'Depth' or depth == 'DEPTH':
            depth = depth.capitalize() + ' (meters)'
            depth_int = 'MD (meters)'
        elif depth == 'MD':
            depth = 'Depth (meters)'
            depth_int = 'MD (meters)'
        elif depth == 'TVDSS':
            depth = 'TVDSS (meters)'
            depth_int = 'TVDSS (meters)'
        elif depth == 'TWT':
            depth = depth_int = depth+' - '+twtcurve+' (ms)'
        else:
            raise ValueError('depth input must be either Depth, TVDSS, or TWT')

        # plot picks
        picks_plot.view(axs, depth=depth, ylim=ylim,
                        labels=labels, pad=pad)

        # plot contacts
        if self.contacts is not None:
            self.contacts.view(axs, depth=depth, ylim=ylim,
                               labels=labels, pad=pad)

        # plot intervals
        if showintervals:
            axs_int = axs[:-1] if skiplast else axs
            for ax in axs_int:
                xlims = ax.get_xlim()
                for index, interval in intervals_plot.df.iterrows():
                    ax.fill_between(xlims, interval['Base '+depth_int],
                                    interval['Top '+depth_int],
                                    facecolor=interval['Color'], alpha=0.2)
                ax.set_xlim(xlims)
        return axs

    def view_logtrack(self, template='petro', lfp=False,
                      depth='MD', twtcurve=None,
                      cali='CALI', gr='GR', rt='RT', vsh='VSH',
                      vcarb='VCARB', vcoal='VCOAL',
                      sgt='SGT', sot='SOT', phi='PHIT',
                      vp='VP', vs='VS', rho='RHOB',
                      ai='AI', vpvs='VPVS', theta=np.arange(0, 40, 5),
                      seismic=None, wav=None, seissampling=1., seisshift=0.,
                      seisreverse=False, trace_in_seismic=False,
                      whichseismic='il', extendseismic=20, thetasub=1,
                      prestack_wiggles=True, horizonset=None,
                      intervals=None, facies=None,
                      faciesfromlog=False, scenario4d=None,
                      title=None, **kwargs_logs):
        """Display log track using any of the provided standard templates
        tailored to different disciplines and analysis

        Parameters
        ----------
        template : :obj:`str`, optional
            Template (``petro``: petrophysical analysis,
            ``rock``: rock-physics analysis,
            ``faciesclass``: facies-classification analysis,
            ``poststackmod``: seismic poststack modelling (with normal
            and averaged properties),
            ``prestackmod``: seismic prestack modelling (with normal
            and averaged properties),
            ``seismic``: 3D seismic intepretation,
            ``prestackseismic``: prestack/AVO seismic intepretation,
            ``4Dmod``: time-lapse seismic modelling)
        lfp : :obj:`bool`, optional
            Prepend `LFP_`` to every log (``True``) or not (``False``)
        depth : :obj:`str`, optional
            Name of depth log curve
        twtcurve : :obj:`str`, optional
            Name of TWT curve to used for y-axis when ``depth=TWT``
        cali : :obj:`str`, optional
            Name of Caliper log
        gr : :obj:`str`, optional
            Name of Gamma Ray log
        rt : :obj:`str`, optional
            Name of Resistivity log
        vsh : :obj:`str`, optional
            Name of gamma ray log
        vcarb : :obj:`str`, optional
            Name of Volume Carbonate log
        vcoal : :obj:`str`, optional
            Name of Volume Coal log
        sgt : :obj:`str`, optional
            Name of Total Gas Saturation Ray log
        sot : :obj:`str`, optional
            Name of Total Oil Saturation log
        phi : :obj:`str`, optional
            Name of Porosity log
        vp : :obj:`str`, optional
            Name of P-wave velocity log
        vs : :obj:`str`, optional
            Name of S-wave velocity log
        vs : :obj:`str`, optional
            Name of S-wave velocity log
        rho : :obj:`str`, optional
            Name of Density log
        ai : :obj:`str`, optional
            Name of Acoustic Impedence log
        vpvs : :obj:`str`, optional
            Name of VP/VS log
        theta : :obj:`np.ndarray`
            Angles in degrees (required for prestack modelling)
        seismic : :obj:`pysubsurface.object.Seismic` or :obj:`pysubsurface.object.SeismicIrregular` or :obj:`pysubsurface.object.SeismicIrregular`, optional
            Name of seismic data to visualize when required by template
            (use ``None`` when not required)
        wav : :obj:`np.ndarray`, optional
            Wavelet to apply to synthetic seismic when required by template
        seissampling : :obj:`float`, optional
            Sampling along depth/time axis for seismic
        seisshift : :obj:`float`, optional
            Shift to apply to real seismic trace. If positive, shift downward,
            if negative shift upward (only available in ``template=seismic``
            or ``template=prestackseismic`` when ``trace_in_seismic=False``)
        seisreverse : :obj:`bool`, optional
            Reverse colors of seismic wavelet filling
        trace_in_seismic : :obj:`bool`, optional
            Display synthetic trace on top of real seismic (``True``) or
            side-by-side with extractec seismic trace (``False``)
        whichseismic : :obj:`str`, optional
            ``il``: display inline section passing through well,
            ``xl``: display crossline section passing through well.
            Note that if well is not vertical an arbitrary path along the well
            trajectory will be chosen
        extendseismic : :obj:`int`, optional
            Number of ilines and crosslines to add at the end of well toe when
            visualizing a deviated well
        thetasub : :obj:`int`, optional
            Susampling factor for angle axis if ``template='prestackseismic'``
            or ``template='prestackmod'``
        prestack_wiggles : :obj:`bool`, optional
            Use wiggles to display pre-stack seismic (``True``) or imshow
            (``False``)
        horizonset : :obj:`dict`, optional
            Horizon set to display if ``template='seismic'``
        intervals : :obj:`int`, optional
            level of intervals to be shown (if ``None``, intervals are not shown)
        facies : :obj:`dict`, optional
            Facies set
        faciesfromlog : :obj:`str`, optional
            Name of log curve with facies (if ``None`` estimate from ``facies``
            definition directly)
        scenario4d : :obj:`str`, optional
            Name of scenario to be used as suffix to select fluid substituted
            well logs for ``template='4D'``
        kwargs_logs : :obj:`dict`, optional
            additional input parameters to be provided to
            :func:`pysubsurface.objects.Logs.visualize_logcurves`

        Returns
        -------
        fig : :obj:`plt.figure`
           Figure handle (``None`` if ``axs`` are passed by user)
        axs : :obj:`plt.axes`
           Axes handles

        """
        # prepend lfp if lfp flag is True
        if lfp:
            cali = 'LFP_'+cali if lfp else cali
            gr = 'LFP_'+gr if lfp else gr
            rt = 'LFP_'+rt if lfp else rt
            vsh = 'LFP_'+vsh if lfp else vsh
            vcarb = 'LFP_'+vcarb if lfp else vcarb
            vcoal = 'LFP_'+vcoal if lfp else vcoal
            sgt = 'LFP_'+sgt if lfp else sgt
            sot = 'LFP_'+sot if lfp else sot
            phi = 'LFP_'+phi if lfp else phi
            vp = 'LFP_'+vp if lfp else vp
            vs = 'LFP_'+vs if lfp else vs
            rho = 'LFP_'+rho if lfp else rho
            ai = 'LFP_'+ai if lfp else ai
            vpvs = 'LFP_'+vpvs if lfp else vpvs

        # define depth for logs
        depthlog = depth + ' - ' + twtcurve if twtcurve is not None else depth

        # plotting
        if template == 'petro':
            fig, axs = \
                self.welllogs.visualize_logcurves(
                    dict(CALI=dict(logs=[cali],
                                   colors=['k'],
                                   xlim=(np.nanmin(self.welllogs.logs[cali]),
                                         np.nanmax(self.welllogs.logs[cali]))),
                         GR=dict(logs=[gr],
                                 colors=['k'],
                                 xlim=(0, np.nanmax(self.welllogs.logs[gr]))),
                         RT=dict(logs=[rt],
                                 colors=['k'],
                                 logscale=True,
                                 xlim=(np.nanmin(self.welllogs.logs[rt]),
                                       np.nanmax(self.welllogs.logs[rt]))),
                         RHOB=dict(logs=[rho],
                                   colors=['k'],
                                   xlim=((np.nanmin(self.welllogs.logs[rho]),
                                          np.nanmax(self.welllogs.logs[rho])))),
                         PHIT=dict(logs=[phi],
                                   colors=['k'],
                                   xlim=(0, 0.4)),
                         Volume=dict(logs=[vsh, vcarb, vcoal],
                                     colors=['green', '#94b8b8',
                                             '#4d4d4d', 'yellow'],
                                     xlim=(0, 1)),
                         Sat=dict(logs=[sgt, sot],
                                  colors=['red', 'green', 'blue'],
                                  envelope=phi,
                                  xlim=(0, 0.4))),
                    depth=depthlog, **kwargs_logs)

        elif template == 'rock':
            fig, axs = \
                self.welllogs.visualize_logcurves(
                    dict(Volume=dict(logs=[vsh, vcarb, vcoal],
                                     colors=['green', '#94b8b8',
                                             '#4d4d4d', 'yellow'],
                                     xlim=(0, 1)),
                         Sat=dict(logs=[sgt, sot],
                                  colors=['red', 'green','blue'],
                                  envelope=phi,
                                  xlim=(0, 0.4)),
                         VP=dict(logs=[vp],
                                 colors=['k'],
                                 xlim=(np.nanmin(self.welllogs.logs[vp]),
                                       np.nanmax(self.welllogs.logs[vp]))),
                         VS=dict(logs=[vs],
                                 colors=['k'],
                                 xlim=(np.nanmin(self.welllogs.logs[vs]),
                                       np.nanmax(self.welllogs.logs[vs]))),
                         RHO=dict(logs=[rho],
                                  colors=['k'],
                                  xlim=(np.nanmin(self.welllogs.logs[rho]),
                                        np.nanmax(self.welllogs.logs[rho]))),
                         AI=dict(logs=[ai],
                                 colors=['k'],
                                 xlim=(np.nanmin(self.welllogs.logs[ai]),
                                       np.nanmax(self.welllogs.logs[ai]))),
                         VPVS=dict(logs=[vpvs],
                                   colors=['k'],
                                   xlim=(np.nanmin(self.welllogs.logs[vpvs]),
                                         np.nanmax(self.welllogs.logs[vpvs])))),
                    depth=depthlog, **kwargs_logs)

        elif template == 'faciesclass':
            figsize = None if 'figsize' not in kwargs_logs.keys() \
                else kwargs_logs['figsize']
            fig, axs = plt.subplots(1, 9, sharey=True, figsize=figsize)
            _, axs = \
                self.welllogs.visualize_logcurves(
                    dict(GR=dict(logs=[gr],
                                 colors=['k'],
                                 xlim=(0, np.nanmax(self.welllogs.logs[gr]))),
                         RT=dict(logs=[rt],
                                 colors=['k'],
                                 logscale=True,
                                 xlim=(np.nanmin(self.welllogs.logs[rt]),
                                       np.nanmax(self.welllogs.logs[rt]))),
                         RHOB=dict(logs=[rho],
                                   colors=['k'],
                                   xlim=((np.nanmin(self.welllogs.logs[rho]),
                                          np.nanmax(self.welllogs.logs[rho])))),
                         PHIT=dict(logs=[phi],
                                   colors=['k'],
                                   xlim=(0, 0.4)),
                         VP=dict(logs=[vp],
                                 colors=['k'],
                                 xlim=(np.nanmin(self.welllogs.logs[vp]),
                                       np.nanmax(self.welllogs.logs[vp]))),
                         VS=dict(logs=[vs],
                                 colors=['k'],
                                 xlim=(np.nanmin(self.welllogs.logs[vs]),
                                       np.nanmax(self.welllogs.logs[vs]))),
                         Volume = dict(logs=[vsh, vcarb, vcoal],
                                       colors=['green', '#94b8b8',
                                               '#4d4d4d', 'yellow'],
                                       xlim=(0, 1)),
                         Sat = dict(logs=[sgt, sot],
                                    colors=['red', 'green', 'blue'],
                                    envelope=phi,
                                    xlim=(0, 0.4))),
                    depth=depthlog, axs=axs, **kwargs_logs)

        elif template == 'poststackmod':
            figsize = None if 'figsize' not in kwargs_logs.keys() \
                else kwargs_logs['figsize']
            fig, axs = plt.subplots(1, 7, sharey=True, figsize=figsize)
            _, axs = \
                self.welllogs.visualize_logcurves(
                    dict(VP=dict(logs=[vp, vp + '_mean'],
                                 colors=['k', '#8c8c8c'],
                                 lw=[2, 8],
                                 xlim=(np.nanmin(self.welllogs.logs[vp]),
                                       np.nanmax(self.welllogs.logs[vp]))),
                         VS=dict(logs=[vs, vs + '_mean'],
                                 colors=['k', '#8c8c8c'],
                                 lw=[2, 8],
                                 xlim=(np.nanmin(self.welllogs.logs[vs]),
                                       np.nanmax(self.welllogs.logs[vs]))),
                         RHO=dict(logs=[rho, rho + '_mean'],
                                  colors=['k', '#8c8c8c'],
                                  lw=[2, 8],
                                  xlim=(np.nanmin(self.welllogs.logs[rho]),
                                        np.nanmax(self.welllogs.logs[rho]))),
                         AI=dict(logs=[ai, ai + '_mean'],
                                 colors=['k', '#8c8c8c'],
                                 lw=[2, 8],
                                 xlim=(np.nanmin(self.welllogs.logs[ai]),
                                       np.nanmax(self.welllogs.logs[ai]))),
                         VPVS=dict(logs=[vpvs, vpvs + '_mean'],
                                   colors=['k', '#8c8c8c'],
                                   lw=[2, 8],
                                   xlim=(np.nanmin(self.welllogs.logs[vpvs]),
                                         np.nanmax(self.welllogs.logs[vpvs]))),
                         Stack=dict(log=ai, sampling=1., wav=wav, title='Modelled Seismic'),
                         Stack1=dict(log=ai + '_mean', sampling=seissampling,
                                     wav=wav, title='Modelled from blocky logs')),
                    depth=depthlog, seisreverse=seisreverse, axs=axs, **kwargs_logs)

        elif template == 'prestackmod':
            figsize = None if 'figsize' not in kwargs_logs.keys() \
                else kwargs_logs['figsize']
            fig, axs = plt.subplots(1, 7, sharey=True, figsize=figsize)
            _, axs = \
                self.welllogs.visualize_logcurves(
                    dict(VP=dict(logs=[vp, vp + '_mean'],
                                 colors=['k', '#8c8c8c'],
                                 lw=[2, 8],
                                 xlim=(np.nanmin(self.welllogs.logs[vp]),
                                       np.nanmax(self.welllogs.logs[vp]))),
                         VS=dict(logs=[vs, vs + '_mean'],
                                 colors=['k', '#8c8c8c'],
                                 lw=[2, 8],
                                 xlim=(np.nanmin(self.welllogs.logs[vs]),
                                       np.nanmax(self.welllogs.logs[vs]))),
                         RHO=dict(logs=[rho, rho + '_mean'],
                                  colors=['k', '#8c8c8c'],
                                  lw=[2, 8],
                                  xlim=(np.nanmin(self.welllogs.logs[rho]),
                                        np.nanmax(self.welllogs.logs[rho]))),
                         AI=dict(logs=[ai, ai + '_mean'],
                                 colors=['k', '#8c8c8c'],
                                 lw=[2, 8],
                                 xlim=(np.nanmin(self.welllogs.logs[ai]),
                                       np.nanmax(self.welllogs.logs[ai]))),
                         VPVS=dict(logs=[vpvs, vpvs + '_mean'],
                                   colors=['k', '#8c8c8c'],
                                   lw=[2, 8],
                                   xlim=(np.nanmin(self.welllogs.logs[vpvs]),
                                         np.nanmax(self.welllogs.logs[vpvs]))),
                         Prestack=dict(theta=theta,
                                       vp=vp,
                                       vs=vs,
                                       rho=rho,
                                       sampling=seissampling,
                                       wav=wav,
                                       scaling=4),
                         Prestack1=dict(theta=theta,
                                        vp=vp + '_mean',
                                        vs=vs + '_mean',
                                        rho=rho + '_mean',
                                        sampling=seissampling,
                                        wav=wav,
                                        scaling=4)),
                    depth=depthlog, seisreverse=seisreverse,
                    prestack_wiggles=prestack_wiggles, axs=axs, **kwargs_logs)

        elif template == 'seismic':
            if not self.vertical:
                raise NotImplementedError('Cannot use template=seismic for non'
                                          'vertical wells')
            if seismic is None or wav is None:
                raise ValueError('Provide a seismic data and a wavelet when '
                                 'visualizing logs with seismic template')

            if trace_in_seismic and depth=='MD':
                trace_in_seismic=False
                logging.warning('Cannot view trace on seismic with depth=MD')

            figsize = None if 'figsize' not in kwargs_logs.keys() else \
                kwargs_logs['figsize']
            if trace_in_seismic:
                fig = plt.figure(figsize=figsize)
                axs = [plt.subplot2grid((1, 7), (0, i)) for i in range(6)]
                axs.append(plt.subplot2grid((1, 7), (0, 5), colspan=2))
            else:
                fig, axs = plt.subplots(1, 7, sharey=True, figsize=figsize)

            logcurves_display = dict(VP=dict(logs=[vp],
                                             colors=['k'],
                                             xlim=(np.nanmin(self.welllogs.logs[vp]),
                                                   np.nanmax(self.welllogs.logs[vp]))),
                                     VS=dict(logs=[vs],
                                             colors=['k'],
                                             xlim=(np.nanmin(self.welllogs.logs[vs]),
                                                   np.nanmax(self.welllogs.logs[vs]))),
                                     RHO=dict(logs=[rho],
                                              colors=['k'],
                                              xlim=(np.nanmin(self.welllogs.logs[rho]),
                                                    np.nanmax(self.welllogs.logs[rho]))),
                                     AI=dict(logs=[ai],
                                         colors=['k'],
                                         xlim=(np.nanmin(self.welllogs.logs[ai]),
                                               np.nanmax(self.welllogs.logs[ai]))),
                                     VPVS=dict(logs=[vpvs],
                                               colors=['k'],
                                               xlim=(np.nanmin(self.welllogs.logs[vpvs]),
                                                     np.nanmax(self.welllogs.logs[vpvs]))))
            if not trace_in_seismic:
                logcurves_display['Stack'] = dict(log=ai,
                                                  sampling=seissampling,
                                                  wav=wav,
                                                  title='Modelled Seismic')
            _, axs = \
                self.welllogs.visualize_logcurves(logcurves_display,
                                                  depth=depthlog,
                                                  axs=axs, seisreverse=seisreverse,
                                                  **kwargs_logs)

            # add real seismic trace
            realtrace = seismic['data'].extract_trace_verticalwell(self)

            if not trace_in_seismic:
                axs[-1] = _wiggletrace(axs[-1],
                                       seismic['data'].tz + seisshift,
                                       realtrace)
                axs[-1].set_xlim(axs[-2].get_xlim())
                axs[-1].set_title('Real Seismic (shift={})'.format(seisshift),
                                  fontsize=12)
            else:
                # find well in il-xl
                ilwell, xlwell = \
                    _findclosest_well_seismicsections(self, seismic['data'],
                                                      traj=False)

                if 'ylim' in kwargs_logs.keys() and \
                        kwargs_logs['ylim'] is not None:
                    axs[-1].set_ylim(kwargs_logs['ylim'])
                axs[-1].set_title('Real Seismic (shift={})'.format(seisshift),
                                  fontsize=12)
                axs[-1].invert_yaxis()

                # find out from first plot and set ylim for all plots
                if 'ylim' in kwargs_logs.keys():
                    zlim_seismic = kwargs_logs['ylim']
                else:
                    zlim_seismic = axs[0].get_ylim()
                for i in range(1, len(axs) - 1):
                    axs[i].set_ylim(zlim_seismic)
                    axs[i].invert_yaxis()
                    axs[i].set_yticks([])
                axs[-1].set_yticks([])

                if horizonset is None:
                    dictseis = {}
                else:
                    dictseis = dict(horizons=horizonset['data'],
                                    horcolors=horizonset['colors'],
                                    horlw=5)
                _, axs[-1] = \
                    self.view_in_seismicsection(seismic['data'], ax=axs[-1],
                                                which=whichseismic,
                                                display_wellname=False,
                                                picks=False,
                                                tzoom_index=False,
                                                tzoom=zlim_seismic,
                                                tzshift=seisshift,
                                                cmap='seismic',
                                                clip=1.,
                                                cbar=True,
                                                interp='sinc',
                                                title='Real Seismic',
                                                **dictseis)
                if whichseismic == 'il':
                    axs[-1].set_xlim(xlwell-extendseismic, xlwell+extendseismic)
                else:
                    axs[-1].set_xlim(ilwell-extendseismic, ilwell+extendseismic)

                trace, zaxisreglog = \
                    zerooffset_wellmod(self.welllogs, depthlog,
                                       seissampling, wav,
                                       ai=ai, zlim=depthlog,
                                       ax=axs[-1])[:2]
                trace_center = xlwell if whichseismic == 'il' else ilwell
                _wiggletrace(axs[-1], zaxisreglog,
                             trace_center + (extendseismic/(4*np.nanmax(trace)))*trace,
                             center=trace_center)

        elif template == 'prestackseismic':
            if seismic is None or wav is None:
                raise ValueError('Provide a prestack seismic data and a wavelet '
                                 'when visualizing logs with seismic template')
            figsize = None if 'figsize' not in kwargs_logs.keys() else \
                kwargs_logs['figsize']
            fig, axs = plt.subplots(1, 7, sharey=True, figsize=figsize)
            _, axs = \
                self.welllogs.visualize_logcurves(
                    dict(VP=dict(logs=[vp],
                                 colors=['k'],
                                 xlim=(np.nanmin(self.welllogs.logs[vp]),
                                       np.nanmax(self.welllogs.logs[vp]))),
                         VS=dict(logs=[vs],
                                 colors=['k'],
                                 xlim=(np.nanmin(self.welllogs.logs[vs]),
                                       np.nanmax(self.welllogs.logs[vs]))),
                         RHO=dict(logs=[rho],
                                  colors=['k'],
                                  xlim=(np.nanmin(self.welllogs.logs[rho]),
                                        np.nanmax(self.welllogs.logs[rho]))),
                         AI=dict(logs=[ai],
                                 colors=['k'],
                                 xlim=(np.nanmin(self.welllogs.logs[ai]),
                                       np.nanmax(self.welllogs.logs[ai]))),
                         VPVS=dict(logs=[vpvs],
                                   colors=['k'],
                                   xlim=(np.nanmin(self.welllogs.logs[vpvs]),
                                         np.nanmax(self.welllogs.logs[vpvs]))),
                         Prestack=dict(theta=theta[::thetasub],
                                       vp=vp,
                                       vs=vs,
                                       rho=rho,
                                       sampling=seissampling,
                                       wav=wav,
                                       scaling=4)),
                    depth=depthlog, seisreverse=seisreverse, axs=axs, **kwargs_logs)

            # add real prestack seismic trace
            realgather = \
                seismic['data'].extract_gather_verticalwell(self, verb=True)
            realgather = realgather[::thetasub]
            axs[-1] = _wiggletracecomb(axs[-1], seismic['data'].tz + seisshift,
                                       theta[::thetasub], realgather,
                                       scaling=20)
            if 'ylim' in kwargs_logs.keys() and kwargs_logs['ylim'] is not None:
                axs[-1].set_ylim(kwargs_logs['ylim'])
            axs[-1].set_title('Real Seismic')
            axs[-1].invert_yaxis()

        elif template == '4Dmod':
            fig, axs = \
                self.welllogs.visualize_logcurves(
                    dict(Volume=dict(logs=[vsh, vcarb, vcoal],
                                     colors=['green', '#94b8b8',
                                             '#4d4d4d', 'yellow'],
                                     xlim=(0, 1)),
                         Sat=dict(logs=[sgt, sot],
                                  colors=['red', 'green','blue'],
                                  envelope=phi,
                                  xlim=(0, 0.4)),
                         Sat1=dict(logs=[sgt+'_'+scenario4d, sot+'_'+scenario4d],
                                  colors=['red', 'green', 'blue'],
                                  envelope=phi,
                                  xlim=(0, 0.4)),
                         AI=dict(logs=[ai, ai+'_'+scenario4d],
                                 colors=['k', 'r'],
                                 lw=[1, 1],
                                 xlim=(np.nanmin(self.welllogs.logs[ai]),
                                       np.nanmax(self.welllogs.logs[ai]))),
                         VPVS=dict(logs=[vpvs, vpvs + '_' + scenario4d],
                                   colors=['k', 'r'],
                                   lw=[1, 1],
                                 xlim=(np.nanmin(self.welllogs.logs[vpvs]),
                                       np.nanmax(self.welllogs.logs[vpvs]))),
                         dAI=dict(logs=[ai+'diff_'+scenario4d],
                                  colors=['k'],
                                  xlim=(-50, 50)),
                         dVPVS=dict(logs=[vpvs+'diff_'+scenario4d],
                                    colors=['k'],
                                    xlim=(-50, 50)),
                         Stack=dict(log=ai,
                                    sampling=seissampling,
                                    wav=wav),
                         Diff=dict(logs=[ai+'_'+scenario4d, ai],
                                   sampling=1.,
                                   wav=wav),
                         PPTimeshift=dict(logs=['PPtimeshift' + '_' + scenario4d],
                                          colors=['k'],
                                          title='PP Timeshift',
                                          xlim=(-1.3 * np.nanmax(np.abs(self.welllogs.logs[
                                                                           'PPtimeshift' + '_' + scenario4d])),
                                                1.3 * np.nanmax(np.abs(self.welllogs.logs[
                                                                          'PPtimeshift' + '_' + scenario4d])))),
                         Prestack=dict(theta=theta,
                                       vp=vp,
                                       vs=vs,
                                       rho=rho,
                                       sampling=seissampling,
                                       wav=wav,
                                       scaling=1.),
                         Prediff=dict(theta=theta,
                                      vp=[vp+'_'+scenario4d, vp],
                                      vs=[vs+'_'+scenario4d, vs],
                                      rho=[rho+'_'+scenario4d, rho],
                                      sampling=seissampling,
                                      wav=wav,
                                      scaling=1.)),
                    depth=depthlog,  seisreverse=seisreverse, ** kwargs_logs)
            xlims = np.array([-np.max(axs[-4].get_xlim()),
                              np.max(axs[-4].get_xlim())])
            axs[-4].set_xlim(xlims)
            axs[-3].set_xlim(xlims)

        else:
            raise ValueError('template={} does not exist'.format(template))

        fig.suptitle(self.wellname+' - '+template if title is None else title,
                     y=0.99, fontsize=20, fontweight='bold')

        if intervals is not None:
            ylim=None if 'ylim' not in kwargs_logs.keys() else \
                kwargs_logs['ylim']
            self.view_picks_and_intervals(axs, depth=depth,
                                          twtcurve=twtcurve,
                                          ylim=ylim, level=intervals,
                                          skiplast=True if template == 'faciesclass'
                                          else False, pad=1.05)

        if template == 'faciesclass':
            xlim_facies = axs[-1].get_xlim()
            faciesnames = list(facies.keys())
            faciescolors = [facies[faciesname].color for faciesname in
                            facies.keys()]

            if faciesfromlog:
                axs[-1] = _visualize_facies(axs[-1], self.welllogs,
                                            faciesfromlog,
                                            faciescolors,
                                            faciesnames,
                                            depth=depth)
            else:
                for faciesname in faciesnames:
                    facies[faciesname].view_on_wellog(self, ax=axs[-1],
                                                      depth=depth,
                                                      xlim=xlim_facies)
                axs[-1].set_xlim(xlim_facies)
        return fig, axs

    def view_in_seismicsection(self, seismic, domain='depth', which='il',
                               onlyinside=False, extend=10, tzshift=0.,
                               surface=None, cmapsurface='gist_rainbow',
                               color='k', display_wellname=True,
                               picks=True, level=2,
                               twtcurve=None, logcurve=None,
                               logcurvethresh=None,
                               logcurvescale=1., logcurveshift=None,
                               logcurvecmap=None, ax=None,
                               **kwargs_seismic):
        """Display IL or XL section of seismic along wellpath.

        Parameters
        ----------
        seismic : :obj:`pysubsurface.objects.Seismic` or :obj:`pysubsurface.objects.SeismicIrregular`
            Seismic to display
        domain : :obj:`str`, optional
            Domain of seismic data, ``depth`` or ``time``
        which : :obj:`str`, optional
            ``il``: display inline section passing through well,
            ``xl``: display crossline section passing through well.
            Note that if well is not vertical an arbitrary path along the well
            trajectory will be chosen
        onlyinside : :obj:`bool`, optional
            Try to infer if well is within seismic limits and display only if
            considered inside (``True``) or display anyhow (``False``)
        extend : :obj:`int`, optional
            Number of ilines and crosslines to add at the end of well toe when
            visualizing a deviated well
        tzshift : :obj:`float`, optional
            Shift to apply to tz axis in seismic
        surface : :obj:`pysubsurface.objects.Surface`, optional
            Surface to visualize with line visualized in seismic section
        cmapsurface : :obj:`str`, optional
            Colormap of surface
        color : :obj:`str`, optional
            Wellpath color
        display_wellname : :obj:`bool`, optional
            Display well name
        picks : :obj:`bool`, optional
            Display picks on well
        level : :obj:`int`
            Interval level to use for display of picks (if ``None`` show all)
        twtcurve : :obj:`str`, optional
            Name of TWT curve as defined in ``self.tdcurve`` or
            ``self.checkshots`` to be used to plot well trajectory when
            seismic is in time domain
        logcurve : :obj:`str`, optional
            Name of log curve to display
        logcurvethresh : :obj:`float`, optional
            Maximum allowed value for log curve (values above set to non-valid)
        logcurvescale : :obj:`float`, optional
            Scaling to apply to log curve
        logcurveshift : :obj:`float`, optional
            Vertical shift to apply to log curve and picks - only for vertical wells
            (if ``None`` no shift is applied)
        logcurvecmap : :obj:`float`, optional
            Cmap to use to fill log curve (if ``None`` do not fill)
        ax : :obj:`plt.axes`
           Axes handle (if ``Ǹone`` make new figure)

        .. note:: This routine cannot find out exactly if the well that you are
          trying to visualize is located inside the coverage of the seismic
          data. Use the ``onlyinside=True`` if you want to ask the routine
          to try to infer and plot only if some checks are verified,
          otherwise provide only wells that are inside the seismic area and set
          ``onlyinside=False``.

        Returns
        -------
        fig : :obj:`plt.figure`
           Figure handle (``None`` if ``axs`` are passed by user)
        ax : :obj:`plt.axes`
           Axes handle

        """
        # choose if well is vertical
        if self._headtoedistance > 400:
            vertical = False
        else:
            vertical = True

        if logcurveshift is None or not vertical:
            logcurveshift = 0.

        # check if welllogs are available, otherwise avoid plotting log curve
        if logcurve is not None:
            if self.welllogs is None:
                logcurve = None
                logging.warning('well {} has no logs...'.format(self.wellname))

        # find IL and XL the well is passing through
        ilwell, xlwell = \
            _findclosest_well_seismicsections(self, seismic,
                                              traj=False if vertical
                                              else True)
        # define trajectories
        trajmd = self.trajectory.df['MD (meters)'].values
        traj_z = self.trajectory.df['TVDSS'].values

        # plot seismic
        if vertical:
            # find out if well may be outside of seismic area
            if ilwell == seismic.ilines[0] or ilwell == seismic.ilines[-1] or \
                    xlwell == seismic.xlines[0] or xlwell == seismic.xlines[-1]:

                    logging.warning('Well {} may be outside of seismic area as '
                                    'closest inline (or crossline) is at the edge '
                                    'of seismic inline (or crossline) '
                                    'axis'.format(self.wellname))
                    if onlyinside:
                        return None, None
            if 'fig' in kwargs_seismic.keys():
                fig = kwargs_seismic['fig']
                kwargs_seismic.pop('fig', None)
            else:
                fig = None
            fig1, ax = \
                seismic.view(ilplot=findclosest(seismic.ilines, ilwell),
                             xlplot=findclosest(seismic.xlines, xlwell),
                             which=which, axs=ax, tzshift=tzshift,
                             **kwargs_seismic)
            if fig is None:
                fig = fig1
            ilwell = ilwell * np.ones(len(traj_z))
            xlwell = xlwell * np.ones(len(traj_z))
            ilwellunique = ilwell
            xlwellunique = xlwell
        else:
            # remove duplicates first from ilines and then from crosslines
            # this is because well trajectory is much more finely sampled
            # than seismic data
            ilxlwellunique, indices = unique_pairs(np.stack([ilwell, xlwell],
                                                             axis=1))
            ilwellunique = ilxlwellunique[:, 0]
            xlwellunique = ilxlwellunique[:, 1]
            traj_z_uniques = traj_z[indices]
            ilwellunique_check = np.unique(ilwellunique)
            xlwellunique_check = np.unique(xlwellunique)

            # find out if well may be outside of seismic area
            if len(ilwellunique_check) == 1 or len(xlwellunique_check) == 1:
                if ilwellunique_check.min() == seismic.ilines[0] or \
                   ilwellunique_check.max() == seismic.ilines[-1] or \
                   xlwellunique_check.min() == seismic.xlines[0] or \
                   xlwellunique_check.max() == seismic.xlines[-1]:
                    logging.warning('Well {} may be outside of seismic area as '
                                    'closest inline (or crossline) is at the edge '
                                    'of seismic inline (or crossline) '
                                    'axis'.format(self.wellname))
                    if onlyinside:
                        return None, None

            # add some ilines and crosslines at the end
            ilwell_seismic = \
                np.hstack((ilwellunique, ilwellunique[-1] +
                           np.sign(ilwellunique[-1]-ilwellunique[0]) * np.arange(extend)))
            xlwell_seismic = \
                np.hstack((xlwellunique,
                           xlwellunique[-1]*np.ones(extend)))

            # remove negative values to avoid wrapping around
            ilxlmask = ilwell_seismic >= 0
            ilwell_seismic = ilwell_seismic[ilxlmask]
            xlwell_seismic = xlwell_seismic[ilxlmask]

            fig, ax = \
                seismic.view_arbitraryline(ilwell_seismic, xlwell_seismic,
                                           tzshift=tzshift, addlines=False,
                                           usevertices=True, jumplabel=20,
                                           ax=ax, **kwargs_seismic)

        # extract time curve and interpolate il and xl to time axis
        if domain == 'time':
            try:
                wellcurve = self.tdcurves[twtcurve].df
            except:
                try:
                    wellcurve = self.checkshots[twtcurve].df
                except:
                    raise ValueError('twtcurve is not present in Well object')

            fil = interp1d(trajmd, ilwellunique,
                         kind='linear',
                         bounds_error=False, assume_sorted=True)
            ilwell = fil(wellcurve['Md (meters)'])
            fxl = interp1d(trajmd, xlwellunique,
                           kind='linear',
                           bounds_error=False, assume_sorted=True)
            xlwell = fxl(wellcurve['Md (meters)'])
            traj_z = wellcurve['Time (ms)'].values

        # extract log depth if logcurve is provided
        if logcurve is not None:
            if domain == 'depth':
                log_z = self.welllogs.df['TVDSS'].values
            else:
                log_z = self.welllogs.df['TWT - '+twtcurve].values

            # interpolate ilwell or xlwell to log sampling
            fil = interp1d(traj_z, ilwell,
                           kind='linear', bounds_error=False,
                           assume_sorted=False)
            ilwelllog = fil(log_z)
            fxl = interp1d(traj_z, xlwell,
                           kind='linear',
                           bounds_error=False, assume_sorted=True)
            xlwelllog = fxl(log_z)

        # plot well trajectory
        if not vertical:
            f = interp1d(ilwellunique, np.arange(len(ilwellunique)),
                         kind='slinear', bounds_error=False)
            ilwellunique = f(ilwell)
            f = interp1d(xlwellunique, np.arange(len(xlwellunique)),
                         kind='slinear', bounds_error=False,
                         assume_sorted=False)
            xlwellunique = f(xlwell)

            traj_x_uniques = np.arange(len(traj_z_uniques))
            traj_x_interp = np.arange(len(traj_z_uniques)*10)
            f = interp1d(traj_x_uniques, traj_z_uniques,
                         kind='slinear', bounds_error=False,
                         assume_sorted=False)
            traj_z_interp = f(traj_x_interp)
        else:
            ilwellunique = ilwell
            xlwellunique = xlwell
            traj_x_interp = ilwellunique if which == 'xl' else xlwellunique
            traj_z_interp = traj_z

        well_text = ~(np.isnan(ilwellunique) | np.isnan(xlwellunique))
        traj_z_text = np.max(traj_z[well_text])
        ilwell_text = ilwellunique[well_text][-1]
        xlwell_text = xlwellunique[well_text][-1]
        ax.plot(traj_x_interp, traj_z_interp, color, lw=2)

        if display_wellname:
            ax.text(1.01 * ilwell_text if not vertical or which == 'xl'
                    else 1.01 * xlwell_text,
                    0.9 * max(traj_z_text, ax.get_ylim()[1]), self.wellname, va="center", color=color,
                    bbox=dict(boxstyle="round", fc=(1., 1., 1.), ec=color))

        # plot log curve
        if logcurve is not None:
            self.welllogs.visualize_logcurve(curve=logcurve,
                                             depth='TVDSS' if domain == 'depth' \
                                             else 'TWT - '+twtcurve,
                                             thresh=logcurvethresh,
                                             shift=xlwelllog if which == 'il' \
                                                 else ilwelllog,
                                             verticalshift=logcurveshift,
                                             scale=logcurvescale,
                                             color='k',
                                             colorcode=False if logcurvecmap is None else True,
                                             cmap=logcurvecmap,
                                             grid=False,
                                             title=None,
                                             xlabelpos=None,
                                             inverty=False,
                                             ax=ax)

        # plot picks and contacts
        if picks and self.intervals is not None:
            if domain == 'depth':
                topdepth = 'Top TVDSS (meters)'
                contactdepth = 'TVDSS (meters)'
            else:
                topdepth = 'Top TWT - '+twtcurve+' (ms)'
                contactdepth = 'TWT (meters)'

            if level is None:
                intervals_plot = self.intervals.copy()
            else:
                if not isinstance(level, (list, tuple)):
                    level = [level]
                intervals_plot = self.intervals[self.intervals['Level'].isin(level)]

            for ipick, pick in intervals_plot.iterrows():
                iclosest = findclosest(traj_z_interp,
                                       float(pick[topdepth]))
                ax.plot(traj_x_interp[iclosest],
                        float(pick[topdepth]) + logcurveshift, marker='_',
                        color=pick['Color'], ms=20, mew=8)
            if self.contacts is not None:
                for icontact, contact in self.contacts.df.iterrows():
                    iclosest = findclosest(traj_z_interp,
                                           float(contact[contactdepth]))
                    ax.plot(traj_x_interp[iclosest],
                            float(contact[contactdepth]) + logcurveshift, marker='_',
                            color=contact['Color'], ms=20, mew=8)

        # plot well perforations and completions
        if self.perforations is not None:
            for perforation in self.perforations.perforations:
                istart = findclosest(traj_z,
                                     perforation.iloc[0]['TVDSS'])
                iend = findclosest(traj_z,
                                   perforation.iloc[-1]['TVDSS'])
                # expand perforation by one/two trajectory elements to allow visualization
                if istart == iend:
                    iend += 2
                elif istart + 1 == iend:
                    iend += 1
                ax.plot(ilwellunique[istart:iend]
                        if not vertical or which == 'xl'
                        else xlwellunique[istart:iend],
                        traj_z[istart:iend], 'k', lw=22)
                ax.plot(ilwellunique[istart:iend]
                        if not vertical or which == 'xl'
                        else xlwellunique[istart:iend],
                        traj_z[istart:iend], 'w', lw=20)

        if self.completions is not None:
            for icomp, completion in enumerate(self.completions.completions):
                istart = findclosest(traj_z,
                                     completion.iloc[0]['TVDSS'])
                iend = findclosest(traj_z,
                                   completion.iloc[-1]['TVDSS'])
                # expand completion by one/two trajectory elements to allow visualization
                if istart == iend:
                    iend += 2
                elif istart + 1 == iend:
                    iend += 1
                c, lw = _completions[
                    self.completions.df.iloc[icomp]['symbol_name']]
                ax.plot(ilwellunique[istart:iend]
                        if not vertical or which == 'xl'
                        else xlwellunique[istart:iend],
                        traj_z[istart:iend], c=c, lw=2 * lw)

        # add surface and line along which seismic is displayed
        if surface is not None:
            axpos = ax.get_position()
            axsurface = fig.add_axes([axpos.x0 + 0.01, axpos.y0 + 0.01,
                                      0.15, 0.15])

            _, axsurface = surface.view(ax=axsurface, cmap=cmapsurface,
                                        originlower=True)
            if vertical:
                if which == 'il':
                    axsurface.axvline(xlwell[0], color='w',
                                      lw=2, linestyle='--')
                else:
                    axsurface.axhline(ilwell[0], color='w',
                                      lw=2, linestyle='--')
            else:
                axsurface.plot(ilwell_seismic, xlwell_seismic, 'w', lw=2)
            axsurface.axis('off')
        return fig, ax

    def view_logprops_intervals(self, level, ax=None, intervals=None,
                                vpname='VP', vsname='VS', rhoname='RHOB',
                                prop1name='VP', prop2name=None, filters=None,
                                draw=False, nsamples=1000,
                                xaxisrev=False, yaxisrev=False,
                                xlim=None, ylim=None,
                                legend=True, title=None):
        """Display histograms (or scatterplots) of log properties
        grouped by intervals for current well

        Parameters
        ----------
        level : :obj:`int`
            Interval level to display
        ax : :obj:`plt.axes`, optional
            Axes handle (if ``None`` draw a new figure)
        intervals : :obj:`str` or :obj:`tuple` , optional
           Interval or Intervals  to display (if ``None`` used entire log)
        vpname : :obj:`str` , optional
           Name of log containing average P-wave velocity
           (needed if ``draw=True``)
        vsname : :obj:`str` , optional
           Name of log containing average S-wave velocity
           (needed if ``draw=True``)
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
        xaxisrev : :obj:`bool`, optional
            Reverse x-axis
        yaxisrev : :obj:`bool`, optional
            Reverse y-axis
        xlim : :obj:`tuple`, optional
            x-axis limits (if ``None`` infer from data)
        ylim : :obj:`tuple`, optional
            y-axis limits (if ``None`` infer from data)
        legend : :obj:`bool`, optional
            Add legend (``True``) or not (``False``)
        title : :obj:`str`, optional
            Title of figure

        Returns
        -------
        fig : :obj:`plt.figure`
           Figure handle (``None`` if ``axs`` are passed by user)
        ax : :obj:`plt.axes`
           Axes handle

        """
        def convert_props(propname, samples):
            if 'AI' in propname:
                samples = samples[0]*samples[2]
            elif 'VPVS' in propname:
                samples = samples[0]/samples[1]
            elif 'VP' in propname:
                samples = samples[0]
            elif 'VS' in propname:
                samples = samples[1]
            elif 'RHOB' in propname:
                samples = samples[2]
            else:
                raise ValueError('propname should contain VP, VS, RHOB, '
                                 'AI or VPVS but it is currently %s' %propname)
            return samples

        # extract level
        if level is None:
            dfinterval = self.intervals[self.intervals['Level'].isnull()]
        else:
            if not isinstance(level, (list, tuple)):
                level = [level]
            dfinterval = self.intervals[self.intervals['Level'].isin(level)]
        if intervals is not None:
            intervals = (intervals) if isinstance(intervals, str) else intervals
            dfinterval = dfinterval[dfinterval['Name'].isin(intervals)]
        # make figure and axes
        if ax is None:
            fig = plt.figure(figsize=(12, 8) if prop2name is None else (12, 8))
            if prop2name is not None:
                ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
                axtab = plt.subplot2grid((3, 1), (2, 0))
            else:
                ax = plt.subplot2grid((2, 1), (0, 0))
                axtab = plt.subplot2grid((2, 1), (1, 0))
        else:
            fig = None

        # loop over intervals and plot values
        for index, interval in dfinterval.iterrows():
            intname = interval['Name']
            intcolor = interval['Color']
            if not draw:
                if prop2name is None:
                    logprop1 = self.extract_logs_in_interval(interval,
                                                             prop1name,
                                                             filters=filters)
                else:
                    logprops = self.extract_logs_in_interval(interval,
                                                             [prop1name,
                                                              prop2name],
                                                             filters=filters)
                    logprop1 = logprops[prop1name]
                    logprop2 = logprops[prop2name]
            else:
                available = (self.averaged_props[vpname][intname]['mean'] is not np.nan) and \
                            (self.averaged_props[vsname][intname]['mean'] is not np.nan) and \
                            (self.averaged_props[rhoname][intname]['mean'] is not np.nan) and \
                            (np.sum(np.isnan(self.averaged_props['Cov'][intname].values)) == 0)

                if available:
                    logprops = \
                        drawsamples(np.array([self.averaged_props[vpname][intname]['mean'],
                                              self.averaged_props[vsname][intname]['mean'],
                                              self.averaged_props[rhoname][intname]['mean']]),
                                    self.averaged_props['Cov'][intname].values, nsamples)
                    logprop1 = convert_props(prop1name, logprops)

                    if prop2name is not None:
                        logprop2 = convert_props(prop2name, logprops)
                else:
                    logprop1 = np.array([])
                    logprop2 = np.array([])

            # plot samples
            if prop2name is None:
                if np.sum(~np.isnan(logprop1))>0:
                    sns.distplot(logprop1[~np.isnan(logprop1)], rug=False,
                                 hist_kws={'color':intcolor, 'alpha': 0.3},
                                 kde_kws={'color': intcolor, 'lw': 3,
                                          'label': intname}, ax=ax)
            else:
                ax.scatter(logprop1, logprop2, s=20,
                           c=interval['Color'], label=intname)
                ax.set_xlabel(prop1name)
                ax.set_ylabel(prop2name)
        if prop2name is not None and legend: ax.legend()
        if title is not None:
            ax.set_title(title)
        if xlim:
            ax.set_xlim(xlim)
        if xaxisrev:
            ax.invert_xaxis()
        if prop2name:
            ax.set_xlabel(prop1name)
            ax.set_ylabel(prop2name)
            if ylim: ax.set_ylim(ylim)
            if yaxisrev: ax.invert_yaxis()

        # add summary table
        if fig is not None:
            axtab.axis("off")
            props = [prop1name+'_mean', prop1name+'_stdev'] if prop2name is None \
                else [prop1name+'_mean', prop1name+'_stdev', prop2name+'_mean',
                      prop2name+'_stdev']
            averaged_props_summary = self.averaged_props_summary[props]
            if intervals is not None:
                averaged_props_summary = averaged_props_summary[averaged_props_summary.index.isin(intervals)]
            axtab.table(cellText=np.round(averaged_props_summary.values,
                                          decimals=2),
                        colWidths=[0.2]*len(props),
                        rowLabels=averaged_props_summary.index,
                        colLabels=averaged_props_summary.columns,
                        bbox=(0.35, 0, 0.4, 1) if prop2name is None
                        else (0, -0.2, 1, 1))
        return fig, ax
