from daqengine.ni import Engine


def main(device):
    '''
    Demonstrates detection of changes on lines configured for hardware-timed
    digital input.
    '''
    def di_callback(names, data):
        print('{} samples from {}'.format(data.shape, names))

    def change_callback(name, change, event_time):
        print('{} edge on {} at {}'.format(change, name, event_time))

    engine = Engine()
    engine.hw_ai_monitor_period = 1
    engine.configure_hw_di(100, '/{}/port0/line0:1'.format(device),
                           names=['poke', 'spout'],
                           clock='/Dev1/Ctr0')
    engine.register_di_callback(di_callback)
    engine.register_di_change_callback(change_callback, debounce=10)
    engine.start()
    raw_input('Demo running. Hit enter to exit.\n')


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
