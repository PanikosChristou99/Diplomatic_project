from pyJoules.device import DeviceFactory
from pyJoules.energy_meter import EnergyMeter
from math import sqrt, tan

devices = DeviceFactory.create_devices()
print(devices)
exit(0)
meter = EnergyMeter(devices)


def foo():
    for i in range(100):
        a = sqrt(float(i))


def bar():
    for i in range(100):
        a = tan(float(i))


meter.start(tag='foo')
foo()
meter.record(tag='bar')
bar()
meter.stop()

trace = meter.get_trace()
