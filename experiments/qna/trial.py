from exptools.core.trial import Trial
import os
import exptools
import json
from psychopy import logging, visual, event
import numpy as np


class QNATrial(Trial):

    def __init__(self, ID=None, parameters = {}, phase_durations = [], session = None, screen = None, tracker = None):

        super(
            QNATrial,
            self).__init__(
            phase_durations=phase_durations,
            session=session,
            screen=screen,
            parameters=parameters,
            tracker=tracker)

        self.sound_stim = self.session.sound_stims[self.parameters['stimulus']]

        self.sound_played = False

    def draw(self, *args, **kwargs):

        self.session.fixation.draw()

        if self.phase == 2:
            if not self.sound_played and self.session.index_number != 0:
                self.sound_stim.play()
                self.sound_played = True

        if (self.phase == 0) and (self.ID == 0):
            self.session.instruction.draw()

        super(QNATrial, self).draw()

    def event(self):

        for ev in event.getKeys():
            if len(ev) > 0:
                if ev in ['esc', 'escape', 'q']:
                    self.events.append(
                        [-99, self.session.clock.getTime() - self.start_time])
                    self.stopped = True
                    self.session.stopped = True
                if ev in ['space', ' ', 't']:
                    if (self.phase == 0) and (self.ID == 0):
                        self.phase_forward()

            super(QNATrial, self).key_event(ev)
