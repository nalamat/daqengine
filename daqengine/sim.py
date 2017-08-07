'''
DAQ device simulator engine
:Author: **Nima Alamatsaz <nima.alamatsaz@njit.edu>**
'''

from __future__ import print_function

import time
import threading
import traceback
import numpy as np

from collections import OrderedDict
from traits.api import HasTraits, Range, List, Bool, Int, on_trait_change, Button, Instance
from traitsui.api import View, Action, Item, CheckListEditor

import logging
log = logging.getLogger(__name__)

class Engine(HasTraits):
    '''
    Simulator interface
    '''
    # Poll period (in seconds). This defines how often callbacks for the analog
    # outputs are notified (i.e., to generate additional samples for playout).
    # If the poll period is too long, then the analog output may run out of
    # samples.
    hw_ao_monitor_period = 1

    # Poll period (in seconds). This defines how quickly acquired (analog input)
    # data is downloaded from the buffers (and made available to listeners). If
    # you want to see data as soon as possible, set the poll period to a small
    # value. If your application is stalling or freezing, set this to a larger
    # value.
    hw_ai_monitor_period = 0.1
    hw_ai2_monitor_period = 0.1

    # Even though data is written to the analog outputs, it is buffered in
    # computer memory until it's time to be transferred to the onboard buffer of
    # the NI acquisition card. NI-DAQmx handles this behind the scenes (i.e.,
    # when the acquisition card needs additional samples, NI-DAQmx will transfer
    # the next chunk of data from the computer memory). We can overwrite data
    # that's been buffered in computer memory (e.g., so we can insert a target
    # in response to a nose-poke). However, we cannot overwrite data that's
    # already been transfered to the onboard buffer. So, the onboard buffer size
    # determines how quickly we can change the analog output in response to an
    # event.
    hw_ao_onboard_buffer = 8191

    # Since any function call takes a small fraction of time (e.g., nanoseconds
    # to milliseconds), we can't simply overwrite data starting at
    # hw_ao_onboard_buffer+1. By the time the function calls are complete, the
    # DAQ probably has already transferred a couple hundred samples to the
    # buffer. This parameter will likely need some tweaking (i.e., only you can
    # determine an appropriate value for this based on the needs of your
    # program).
    hw_ao_min_writeahead = hw_ao_onboard_buffer + 1000

    hw_di_names = List
    hw_di_state = List

    sw_do_names = List
    sw_do_state = List

    def __init__(self):
        # Use an OrderedDict to ensure that when we loop through the tasks
        # stored in the dictionary, we process them in the order they were
        # configured.
        self._tasks = OrderedDict()

        self._view = View(
            Item('hw_di_state', label='Digital input' , editor=CheckListEditor(name='hw_di_names',cols=8), style='custom'),
            Item('sw_do_state', label='Digital output', editor=CheckListEditor(name='sw_do_names',cols=8), style='custom'),
            title='DAQ Engine Simulator',
            resizable=True,
            kind='live'
        )

    @on_trait_change('hw_di_state')
    def _hw_di_state_changed(self, name, old, new):
        pass

    def configure_hw_ao(self, fs, lines, expected_range, names=None,
                        start_trigger=None, timebase_src=None, timebase_rate=None):
        callback_samples = int(self.hw_ao_monitor_period*fs)
        thread = threading.Thread(target=self._thread_loop_hw_ao, args=[])
        task = {'thread':thread, 'thread_stop':False, 'fs':fs, 'names':names,
            'callbacks':[], 'callback_samples':callback_samples}
        self._tasks['hw_ao'] = task

    def configure_hw_ai(self, fs, lines, expected_range, names=None,
                        start_trigger=None, timebase_src=None, timebase_rate=None):
        callback_samples = int(self.hw_ai_monitor_period*fs)
        thread = threading.Thread(target=self._thread_loop_hw_ai, args=[])
        task = {'thread':thread, 'thread_stop':False, 'fs':fs, 'names':names,
            'callbacks':[], 'callback_samples':callback_samples}
        self._tasks['hw_ai'] = task

    def configure_hw_ai2(self, fs, lines, expected_range, names=None,
                        start_trigger=None, timebase_src=None, timebase_rate=None):
        callback_samples = int(self.hw_ai2_monitor_period*fs)
        thread = threading.Thread(target=self._thread_loop_hw_ai2, args=[])
        task = {'thread':thread, 'thread_stop':False, 'fs':fs, 'names':names,
            'callbacks':[], 'callback_samples':callback_samples}
        self._tasks['hw_ai2'] = task

    def configure_hw_di(self, fs, lines, names=None, start_trigger=None, clock=None):
        callback_samples = int(self.hw_ai_monitor_period*fs)
        thread = threading.Thread(target=self._thread_loop_hw_di, args=[])
        task = {'thread':thread, 'thread_stop':False, 'fs':fs, 'names':names,
            'callbacks':[], 'callback_samples':callback_samples}
        self._tasks['hw_di'] = task
        self.hw_di_names = names

    def configure_sw_do(self, lines, names=None, initial_state=None):
        if initial_state is None:
            initial_state = np.zeros(len(names), dtype=np.uint8)
        task = {'names':names, 'state':initial_state}
        self._tasks['sw_do'] = task
        self.sw_do_names = names


    def register_ao_callback(self, callback):
        self._tasks['hw_ao']['callbacks'].append(callback)

    def register_ai_callback(self, callback):
        self._tasks['hw_ai']['callbacks'].append(callback)

    def register_ai2_callback(self, callback):
        self._tasks['hw_ai']['callbacks'].append(callback)

    def register_di_change_callback(self, callback, debounce=1):
        self._tasks['hw_di']['callbacks'].append(callback)


    def write_hw_ao(self, data, offset=None):
        pass
        # task = self._tasks['hw_ao']
        # if offset is not None:
        #     # Overwrites data already in the buffer. Used to override changes to
        #     # the signal.
        #     mx.DAQmxSetWriteRelativeTo(task, mx.DAQmx_Val_FirstSample)
        #     mx.DAQmxSetWriteOffset(task, offset)
        #     log.trace('Writing %d samples starting at %d', data.size, offset)
        # else:
        #     # Appends data to the end of the buffer.
        #     mx.DAQmxSetWriteRelativeTo(task, mx.DAQmx_Val_CurrWritePos)
        #     mx.DAQmxSetWriteOffset(task, 0)
        #     log.trace('Writing %d samples to end of buffer', data.size)
        # mx.DAQmxWriteAnalogF64(task, data.shape[-1], False, 0,
        #                        mx.DAQmx_Val_GroupByChannel,
        #                        data.astype(np.float64), self._int32, None)
        # if self._int32.value != data.shape[-1]:
        #     raise ValueError('Unable to write all samples to channel')

    def write_sw_do(self, state):
        task = self._tasks['sw_do']
        state = np.asarray(state).astype(np.uint8)
        task['state'] = state

    def set_sw_do(self, name, state):
        task = self._tasks['sw_do']
        i = task['names'].index(name)
        task['state'][i] = state


    def _thread_loop_hw_ao(self):
        try:
            task = self._tasks['hw_ao']
            samples = task['callback_samples']
            interval = samples / task['fs']
            counter = 0
            while not task['thread_stop']:
                t = time.time()
                if t-self.start_time>=counter*interval:
                    offset = counter*interval*task['fs']
                    for cb in task['callbacks']:
                        cb(task['names'], offset, samples)
                    counter += 1
                time.sleep(0.001)
        except:
            print(traceback.format_exc())
            print('Exiting _thread_loop_hw_ao')

    def _thread_loop_hw_ai(self):
        try:
            task = self._tasks['hw_ai']
            channels = len(task['names'])
            samples = task['callback_samples']
            interval = samples / task['fs']
            counter = 1
            while not task['thread_stop']:
                t = time.time()
                if t-self.start_time>=counter*interval:
                    ts = np.arange(counter*interval,(counter+1)*interval,1/task['fs'])
                    data = np.zeros([channels, len(ts)], dtype=np.double)
                    for i in range(channels):
                        data[i,:] = np.sin(2*np.pi*1*ts + i*2*np.pi/channels)
                    for cb in task['callbacks']:
                        cb(task['names'], data)
                    counter += 1
                time.sleep(0.001)
        except:
            print(traceback.format_exc())
            print('Exiting _thread_loop_hw_ai')

    def _thread_loop_hw_ai2(self):
        try:
            task = self._tasks['hw_ai2']
            channels = len(task['names'])
            samples = task['callback_samples']
            interval = samples / task['fs']
            counter = 1
            while not task['thread_stop']:
                t = time.time()
                if t-self.start_time>=counter*interval:
                    data = np.random.rand(channels,samples)
                    for cb in task['callbacks']:
                        cb(task['names'], data)
                    counter += 1
                time.sleep(0.001)
        except:
            print(traceback.format_exc())
            print('Exiting _thread_loop_hw_ai2')

    def _thread_loop_hw_di(self):
        try:
            task = self._tasks['hw_di']
            while not task['thread_stop']:
                # for cb in task['callbacks']:
                #     cb(name, change, timestamp)
                time.sleep(0.001)
        except:
            print(traceback.format_exc())
            print('Exiting _thread_loop_hw_di')


    def start(self):
        self.start_time = time.time()
        for task in self._tasks.values():
            if 'thread' in task:
                task['thread'].start()
        self.configure_traits(view=self._view)

    def stop(self):
        for task in self._tasks.values():
            if 'thread' in task:
                task['thread_stop'] = True
                task['thread'].join()


    def ao_sample_clock(self):
        elapsed_time = time.time() - self.start_time
        return elapsed_time * self._tasks['hw_ao']['fs']

    def ao_sample_clock_rate(self):
        return self._tasks['hw_ao']['fs']

    def ai_sample_clock(self):
        elapsed_time = time.time() - self.start_time
        return elapsed_time * self._tasks['hw_ai']['fs']

    def ai_sample_clock_rate(self):
        return self._tasks['hw_ai']['fs']
