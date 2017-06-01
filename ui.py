# coding: utf-8
import time

import sys
import itertools

from pip.utils.ui import DownloadProgressBar, DownloadProgressSpinner
from pip.utils.ui import _select_progress_class, WindowsMixin, InterruptibleMixin, DownloadProgressMixin, WritelnMixin
from pip._vendor.progress.bar import Bar, IncrementalBar, ShadyBar, ChargingBar, FillingSquaresBar, FillingCirclesBar
from pip._vendor.progress.spinner import Spinner, PieSpinner, MoonSpinner, LineSpinner

_BaseBar = _select_progress_class(ShadyBar, Bar)
class DownloadProgressBarShady(WindowsMixin, InterruptibleMixin,
                          DownloadProgressMixin, _BaseBar):
    file = sys.stdout
    message = "%(percent)d%%"
    suffix = "%(downloaded)s %(download_speed)s %(pretty_eta)s"

_BaseBar = _select_progress_class(ChargingBar, Bar)
class DownloadProgressBarCharging(WindowsMixin, InterruptibleMixin,
                          DownloadProgressMixin, _BaseBar):
    file = sys.stdout
    message = "%(percent)d%%"
    suffix = "%(downloaded)s %(download_speed)s %(pretty_eta)s"

_BaseBar = _select_progress_class(FillingSquaresBar, Bar)
class DownloadProgressBarFillingSquares(WindowsMixin, InterruptibleMixin,
                          DownloadProgressMixin, _BaseBar):
    file = sys.stdout
    message = "%(percent)d%%"
    suffix = "%(downloaded)s %(download_speed)s %(pretty_eta)s"

_BaseBar = _select_progress_class(FillingCirclesBar, Bar)
class DownloadProgressBarFillingCircles(WindowsMixin, InterruptibleMixin,
                          DownloadProgressMixin, _BaseBar):
    file = sys.stdout
    message = "%(percent)d%%"
    suffix = "%(downloaded)s %(download_speed)s %(pretty_eta)s"

class DownloadProgressSpinner(WindowsMixin, InterruptibleMixin,
                              DownloadProgressMixin, WritelnMixin, Spinner):
    file = sys.stdout
    suffix = "%(downloaded)s %(download_speed)s"

    def next_phase(self):
        if not hasattr(self, "_phaser"):
            self._phaser = itertools.cycle(self.phases)
        return next(self._phaser)

    def update(self):
        message = self.message % self
        phase = self.next_phase()
        suffix = self.suffix % self
        line = ''.join([
            message,
            " " if message else "",
            phase,
            " " if suffix else "",
            suffix,
        ])

        self.writeln(line)

class DownloadProgressSpinnerMoon(WindowsMixin, InterruptibleMixin,
                              DownloadProgressMixin, WritelnMixin, MoonSpinner):

    file = sys.stdout
    suffix = "%(downloaded)s %(download_speed)s"

    def next_phase(self):
        if not hasattr(self, "_phaser"):
            self._phaser = itertools.cycle(self.phases)
        return next(self._phaser)

    def update(self):
        message = self.message % self
        phase = self.next_phase()
        suffix = self.suffix % self
        line = ''.join([
            message,
            " " if message else "",
            phase,
            " " if suffix else "",
            suffix,
        ])

        self.writeln(line)

class DownloadProgressSpinnerPie(WindowsMixin, InterruptibleMixin,
                              DownloadProgressMixin, WritelnMixin, PieSpinner):

    file = sys.stdout
    suffix = "%(downloaded)s %(download_speed)s"

    def next_phase(self):
        if not hasattr(self, "_phaser"):
            self._phaser = itertools.cycle(self.phases)
        return next(self._phaser)

    def update(self):
        message = self.message % self
        phase = self.next_phase()
        suffix = self.suffix % self
        line = ''.join([
            message,
            " " if message else "",
            phase,
            " " if suffix else "",
            suffix,
        ])

        self.writeln(line)

class DownloadProgressSpinnerLine(WindowsMixin, InterruptibleMixin,
                              DownloadProgressMixin, WritelnMixin, LineSpinner):

    file = sys.stdout
    suffix = "%(downloaded)s %(download_speed)s"

    def next_phase(self):
        if not hasattr(self, "_phaser"):
            self._phaser = itertools.cycle(self.phases)
        return next(self._phaser)

    def update(self):
        message = self.message % self
        phase = self.next_phase()
        suffix = self.suffix % self
        line = ''.join([
            message,
            " " if message else "",
            phase,
            " " if suffix else "",
            suffix,
        ])

        self.writeln(line)
