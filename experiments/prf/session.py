from __future__ import division
from psychopy import visual, core, misc, event, data
import numpy as np
# from IPython import embed as shell
from math import *

import os
import sys
import time
import pickle
import pygame
from pygame.locals import *

from exptools.core.session import EyelinkSession

from trial import *
from stim import *

import appnope
appnope.nope()


class PRFSession(EyelinkSession):
    def __init__(self, subject_initials, index_number, scanner, tracker_on, **kwargs):
        super(PRFSession, self).__init__(subject_initials=subject_initials, index_number=index_number, tracker_on=tracker_on, **kwargs)

        screen = self.create_screen()
        # screen = self.create_screen( size = screen_res, full_screen =0, physical_screen_distance = 159.0, background_color = background_color, physical_screen_size = (70, 40) )
        event.Mouse(visible=False, win=screen)

        self.create_output_filename()
        if tracker_on:
            # self.create_tracker(auto_trigger_calibration = 1, calibration_type = 'HV9')
            # if self.tracker_on:
            #     self.tracker_setup()
           # how many points do we want:
            n_points = 9

            # order should be with 5 points: center-up-down-left-right
            # order should be with 9 points: center-up-down-left-right-leftup-rightup-leftdown-rightdown
            # order should be with 13: center-up-down-left-right-leftup-rightup-leftdown-rightdown-midleftmidup-midrightmidup-midleftmiddown-midrightmiddown
            # so always: up->down or left->right

            # creat tracker
            self.create_tracker(auto_trigger_calibration=0,
                                calibration_type='HV%d' % n_points)

            # it is setup to do a 9 or 5 point circular calibration, at reduced ecc

            # create 4 x levels:
            # width = self.eyelink_calib_size * DISPSIZE[1]
            # x_start = (DISPSIZE[0]-width)/2
            # x_end = DISPSIZE[0]-(DISPSIZE[0]-width)/2
            # x_range = np.linspace(x_start, x_end, 5) + \
            #     self.x_offset
            # y_start = (DISPSIZE[1]-width)/2
            # y_end = DISPSIZE[1]-(DISPSIZE[1]-width)/2
            # y_range = np.linspace(y_start, y_end, 5)

            # # set calibration targets
            # cal_center = [x_range[2], y_range[2]]
            # cal_left = [x_range[0], y_range[2]]
            # cal_right = [x_range[4], y_range[2]]
            # cal_up = [x_range[2], y_range[0]]
            # cal_down = [x_range[2], y_range[4]]
            # cal_leftup = [x_range[1], y_range[1]]
            # cal_rightup = [x_range[3], y_range[1]]
            # cal_leftdown = [x_range[1], y_range[3]]
            # cal_rightdown = [x_range[3], y_range[3]]

            # # create 4 x levels:
            # width = self.eyelink_calib_size * \
            #     0.75 * DISPSIZE[1]
            # x_start = (DISPSIZE[0]-width)/2
            # x_end = DISPSIZE[0]-(DISPSIZE[0]-width)/2
            # x_range = np.linspace(x_start, x_end, 5) + \
            #     self.x_offset
            # y_start = (DISPSIZE[1]-width)/2
            # y_end = DISPSIZE[1]-(DISPSIZE[1]-width)/2
            # y_range = np.linspace(y_start, y_end, 5)

            # # set calibration targets
            # val_center = [x_range[2], y_range[2]]
            # val_left = [x_range[0], y_range[2]]
            # val_right = [x_range[4], y_range[2]]
            # val_up = [x_range[2], y_range[0]]
            # val_down = [x_range[2], y_range[4]]
            # val_leftup = [x_range[1], y_range[1]]
            # val_rightup = [x_range[3], y_range[1]]
            # val_leftdown = [x_range[1], y_range[3]]
            # val_rightdown = [x_range[3], y_range[3]]

            # # get them in the right order
            # if n_points == 5:
            #     cal_xs = np.round(
            #         [cal_center[0], cal_up[0], cal_down[0], cal_left[0], cal_right[0]])
            #     cal_ys = np.round(
            #         [cal_center[1], cal_up[1], cal_down[1], cal_left[1], cal_right[1]])
            #     val_xs = np.round(
            #         [val_center[0], val_up[0], val_down[0], val_left[0], val_right[0]])
            #     val_ys = np.round(
            #         [val_center[1], val_up[1], val_down[1], val_left[1], val_right[1]])
            # elif n_points == 9:
            #     cal_xs = np.round([cal_center[0], cal_up[0], cal_down[0], cal_left[0], cal_right[0],
            #                        cal_leftup[0], cal_rightup[0], cal_leftdown[0], cal_rightdown[0]])
            #     cal_ys = np.round([cal_center[1], cal_up[1], cal_down[1], cal_left[1], cal_right[1],
            #                        cal_leftup[1], cal_rightup[1], cal_leftdown[1], cal_rightdown[1]])
            #     val_xs = np.round([val_center[0], val_up[0], val_down[0], val_left[0], val_right[0],
            #                        val_leftup[0], val_rightup[0], val_leftdown[0], val_rightdown[0]])
            #     val_ys = np.round([val_center[1], val_up[1], val_down[1], val_left[1], val_right[1],
            #                        val_leftup[1], val_rightup[1], val_leftdown[1], val_rightdown[1]])
            # #xs = np.round(np.linspace(x_edge,DISPSIZE[0]-x_edge,n_points))
            # #ys = np.round([self.ywidth/3*[1,2][pi%2] for pi in range(n_points)])

            # # put the points in format that eyelink wants them, which is
            # # calibration_targets / validation_targets: 'x1,y1 x2,y2 ... xz,yz'
            # calibration_targets = ' '.join(
            #     ['%d,%d' % (cal_xs[pi], cal_ys[pi]) for pi in range(n_points)])
            # # just copy calibration targets as validation for now:
            # #validation_targets = calibration_targets
            # validation_targets = ' '.join(
            #     ['%d,%d' % (val_xs[pi], val_ys[pi]) for pi in range(n_points)])

            # # point_indices: '0, 1, ... n'
            # point_indices = ', '.join(['%d' % pi for pi in range(n_points)])

            # # and send these targets to the custom calibration function:
            # self.custom_calibration(calibration_targets=calibration_targets,
            #                         validation_targets=validation_targets, point_indices=point_indices,
            #                         n_points=n_points, randomize_order=True, repeat_first_target=True,)
            # reapply settings:
            self.tracker_setup()
        else:
            self.create_tracker(tracker_on=False)

        self.response_button_signs = dict(zip(self.config.get('buttons', 'keys'), range(len(self.config.get('buttons', 'keys')))))

        self.scanner = scanner
        # trials can be set up independently of the staircases that support their parameters
        self.prepare_trials(**kwargs)

    def prepare_trials(self, **kwargs):
        """docstring for prepare_trials(self):"""

        self.directions = np.linspace(0, 2.0 * pi, 8, endpoint=False)
        # Set arguments from config file or kwargs
        for argument in ['mask_type', 'vertical_stim_size', 'horizontal_stim_size',
                         'bar_width_ratio', 'num_elements', 'color_ratio', 'element_lifetime',
                         'stim_present_booleans', 'stim_direction_indices',
                         'fixation_outer_rim_size', 'fixation_rim_size', 'fixation_size',
                         'fast_speed', 'slow_speed', 'element_size', 'element_spatial_frequency']:
            value = kwargs.pop(argument, self.config.get('stimuli', argument))
            setattr(self, argument, value)

        if self.mask_type == 0:
            self.horizontal_stim_size = self.size[1]/self.size[0]

         # Set arguments from config file or kwargs
        for argument in ['PRF_ITI_in_TR', 'TR', 'task_rate', 'task_rate_offset',
                         'vertical_bar_pass_in_TR', 'horizontal_bar_pass_in_TR', 'empty_bar_pass_in_TR']:
            value = kwargs.pop(argument, self.config.get('timing', argument))
            setattr(self, argument, value)
        # orientations, bar moves towards:
        # 0: S      3: NW   6: E
        # 1: SW     4: N    7: SE
        # 2: W      5: NE

        self.bar_pass_durations = []
        for i in range(len(self.stim_present_booleans)):
            if self.stim_present_booleans[i] == 0:
                self.bar_pass_durations.append(
                    self.empty_bar_pass_in_TR * self.TR)
            else:
                if self.stim_direction_indices[i] in (2, 6):  # EW-WE:
                    self.bar_pass_durations.append(
                        self.horizontal_bar_pass_in_TR * self.TR)
                elif self.stim_direction_indices[i] in (0, 4):  # NS-SN:
                    self.bar_pass_durations.append(
                        self.vertical_bar_pass_in_TR * self.TR)

        # nostim-top-left-bottom-right-nostim-top-left-bottom-right-nostim
        # nostim-bottom-left-nostim-right-top-nostim
        self.trial_array = np.array(
            [[self.stim_direction_indices[i], self.stim_present_booleans[i]] for i in range(len(self.stim_present_booleans))])

        self.RG_color = 1/self.color_ratio
        self.BY_color = 1

        self.phase_durations = np.array([[
            -0.001,  # instruct time
            180.0,  # wait for scan pulse
            self.bar_pass_durations[i],
            self.PRF_ITI_in_TR * self.TR] for i in range(len(self.stim_present_booleans))])    # ITI


        self.total_duration = np.sum(np.array(self.phase_durations))
        self.phase_durations[0,0] = 1800

        # fixation point
        self.fixation_outer_rim = visual.GratingStim(self.screen, mask='raisedCos', tex=None, size=self.deg2pix(self.fixation_outer_rim_size),
                                                   pos=np.array((self.x_offset, 0.0)), color=self.background_color, maskParams={'fringeWidth': 0.4})
        self.fixation_rim = visual.GratingStim(self.screen, mask='raisedCos', tex=None, size=self.deg2pix(self.fixation_rim_size),
                                             pos=np.array((self.x_offset, 0.0)), color=(-1.0, -1.0, -1.0), maskParams={'fringeWidth': 0.4})
        self.fixation = visual.GratingStim(self.screen, mask='raisedCos', tex=None, size=self.deg2pix(self.fixation_size),
                                         pos=np.array((self.x_offset, 0.0)), color=self.background_color, opacity=1.0, maskParams={'fringeWidth': 0.4})

        # mask
        if self.mask_type == 1:
            draw_screen_space = [self.screen_pix_size[0]*self.horizontal_stim_size,
                                 self.screen_pix_size[1]*self.vertical_stim_size]
            mask = np.ones(
                (self.screen_pix_size[1], self.screen_pix_size[0]))*-1
            x_edge = int(
                np.round((self.screen_pix_size[0]-draw_screen_space[0])/2))
            y_edge = int(
                np.round((self.screen_pix_size[1]-draw_screen_space[1])/2))
            if x_edge > 0:
                mask[:, :x_edge] = 1
                mask[:, -x_edge:] = 1
            if y_edge > 0:
                mask[-y_edge:, :] = 1
                mask[:y_edge, :] = 1
            import scipy
            mask = scipy.ndimage.filters.gaussian_filter(mask, 5)
            self.mask_stim = visual.GratingStim(self.screen, mask=mask, tex=None, size=[self.screen_pix_size[0], self.screen_pix_size[1]],
                                              pos=np.array((self.x_offset, 0.0)), color=self.screen.background_color)
        elif self.mask_type == 0:
            mask = filters.makeMask(matrixSize=self.screen_pix_size[0], shape='raisedCosine', radius=self.vertical_stim_size *
                                    self.screen_pix_size[1]/self.screen_pix_size[0]/2, center=(0.0, 0.0), range=[1, -1], fringeWidth=0.1)
            self.mask_stim = visual.GratingStim(self.screen, mask=mask, tex=None, 
                size=[self.screen_pix_size[0]*2, self.screen_pix_size[0]*2], 
                pos=np.array((self.x_offset, 0.0)), 
                color=self.screen.background_color)

        # fixation task timing
        self.fix_task_frame_values = self._get_frame_values(framerate=self.framerate, 
                                trial_duration=self.total_duration, 
                                min_value=1.0,
                                exp_scale=1.0,
                                safety_margin=3000.0)


    def run(self):
        """docstring for fname"""
        # cycle through trials
        for i in range(len(self.trial_array)):
            # prepare the parameters of the following trial based on the shuffled trial array
            this_trial_parameters = {}
            this_trial_parameters['stim_duration'] = self.phase_durations[i, -2]
            this_trial_parameters['orientation'] = self.directions[self.trial_array[i, 0]]
            this_trial_parameters['stim_bool'] = self.trial_array[i, 1]

            # these_phase_durations = self.phase_durations.copy()
            these_phase_durations = self.phase_durations[i]

            this_trial = PRFTrial(this_trial_parameters, phase_durations=these_phase_durations,
                                  session=self, screen=self.screen, tracker=self.tracker)

            # run the prepared trial
            this_trial.run(ID=i)
            if self.stopped == True:
                break
        self.close()

    def _get_frame_values(self,
                          framerate,
                          trial_duration,
                          min_value,
                          exp_scale,
                          values=[-1, 1],
                          safety_margin=None):

        if safety_margin is None:
            safety_margin = 5

        n_values = len(values)

        total_duration = trial_duration + safety_margin
        total_n_frames = total_duration * framerate

        result = np.zeros(int(total_n_frames))

        n_samples = np.ceil(total_duration * 2 /
                            (exp_scale + min_value)).astype(int)
        durations = np.random.exponential(exp_scale, n_samples) + min_value

        frame_times = np.linspace(
            0, total_duration, total_n_frames, endpoint=False)

        first_index = np.random.randint(n_values)

        result[frame_times < durations[0]] = values[first_index]

        for ix, c in enumerate(np.cumsum(durations)):
            result[frame_times > c] = values[(first_index + ix) % n_values]

        return result
