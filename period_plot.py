import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
import astropy.units as u
import astropy.constants as const
from astropy.modeling import models, fitting
from astropy.time import Time
from datetime import date
from astropy.timeseries import LombScargle
import matplotlib.gridspec as gridspec


data = Table.read('gj1132_lya_lc.ecsv')
times, flux, error = np.array(data['MJD']), np.array(data['FLUX']), np.array(data['ERROR'])

fitter = fitting.LevMarLSQFitter()

plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1,7)
ls =  LombScargle(times, flux, dy=np.array(error), normalization='model')
frequency, power = ls.autopower(maximum_frequency = 1/100, minimum_frequency=1/1500, samples_per_peak=10)
plt.subplot(gs[0:3])

plt.plot(1/frequency, power)
period = 1/frequency[np.argmax(power)]
print(period)
plt.xlabel('Period (d)')
plt.ylabel('LS Power')
#plt.annotate('P\_max = {0:10.1f} d'.format(period), (0.6, 0.9), xycoords ='axes fraction' )
#plt.axhline(ls.false_alarm_level(0.01))
plt.xlim(101, 1399)


plt.subplot(gs[3:])
plt.errorbar(times, flux, yerr=error, ls='none', marker='o')

plt.xlabel('Time (MJD)')
plt.ylabel(r'Flux$_{Ly\alpha}$ (erg s$^{-1}$ cm$^{-2}$)')

#plt.ylim(-1e-12, 2.8e-12)
today = Time('2020-3-03', format='iso')
today = today.mjd
#print(today)
#plt.axvline(today+21, ls='-.', c='C0', label='Requested observing window ')
#plt.axvspan(today+2, today+31, color='C0', alpha=0.5)
#plt.axvspan(today+85, today+95, color='C0', alpha=0.5)
plt.axvline(today)
#plt.axvline(59000, ls = '-.', c='C0')
#plt.axvline(58950, ls = '-.', c='C0')



timefit = np.arange(times[0]-100, 59488, 10)

#f_guess = frequency[np.argmax(power)]
#f_guess = 1/550
periods = []
p_errors = []
for f_guess in (frequency[np.argmax(power)], 1/550, 1/1000, 1/130):
    sin_mod =  models.Sine1D(amplitude=1e-12, frequency=f_guess) + models.Const1D(2e-12)
    sin_fit = fitter(sin_mod, times, flux, weights = 1/np.array(error), maxiter=100000)
    sin_fit_e = np.sqrt(np.diag(fitter.fit_info['param_cov']))
    
    p_fit =  1/sin_fit[0].frequency
    periods.append(p_fit)
    p_e = sin_fit_e[1]/(sin_fit[0].frequency.value**2)
    p_errors.append(p_e)
    print(p_fit)
    print(p_e)
    label = 'P = {0:10.1f} d '.format(p_fit)
    idx = (np.abs(timefit - today)).argmin()
    print(sin_fit(timefit)[idx])
    #label = 'P = {0:10.0f} $\pm$ {1:10.0f}d '.format(p_fit, p_e)
    
    plt.plot(timefit, sin_fit(timefit), label=label, ls='--')
#plt.annotate('P\_fit = {0:10.1f} $\pm$ {1:10.1f}d '.format(p_fit, p_e), (0.5, 0.9), xycoords ='axes fraction' )
#plt.xlim(times[0]-300, today+2000)
plt.ylim(0.1e-12, 4.1e-12)
plt.xlim(timefit[0], timefit[-1])
plt.legend(loc=2)


plt.subplot(gs[0:3])
[plt.axvline(p, ls='--', c=c) for p, c in zip(periods,['C1', 'C2', 'C3', 'C4'])]

plt.tight_layout()
#plt.savefig('four_periods.pdf', dpi=100)
#print(len(flux))
plt.show()