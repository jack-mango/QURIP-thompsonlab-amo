import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from .imagestack_functions import scaleImageStack, getCounts, getOccupancy, getBkgSubCounts, findTweezerPositions_v2


class weighted_histogram_556:
    """
    The class usually used in acquiring the thresholds and weights for the green (556 nm) imaging. Tha data put in
    is designed to be from a continous imaging experiment, with each image stack of shape [nLoops * imagesTakenPerLoop,
    picWidth, picHeight].
    """

    def __init__(self, files, nLoops, nTweezers=None, positions=None, rAtom=2, exclude1st=True):
        self.cameraType = 'pvcam'
        try:
            iter(files)
        except:  # single file
            self.imageStack = scaleImageStack(files.load(), cameraType=self.cameraType)
        else:  # multiple files
            dum = []
            for i in np.arange(np.shape(files)[0]):
                imgs = scaleImageStack(files[i].load(), cameraType=self.cameraType)
                imgs2 = np.reshape(imgs, (-1, nLoops, np.shape(imgs)[1], np.shape(imgs)[2]))
                if exclude1st:
                    dum.append(imgs2[1:, :, :, :])
                else:
                    dum.append(imgs2[:, :, :, :])
            # 'dum' now is of shape [nFiles, nImagesPerSequence, nLoops, len_x, len_y] (we often transpose, so the x, y
            # are reversed here comparing to on the PVCAM GUI.)
            self.imageStack = np.reshape(dum, (-1, np.shape(dum)[3], np.shape(dum)[4]))

        self.files = files
        self.nLoops = nLoops
        self.exclude1st = exclude1st
        self.nImages = np.shape(self.imageStack)[0]
        self.summedImage = self.imageStack.sum(axis=0)
        self.pixelWeight = None
        self.thresholds = None
        self.rAtom = rAtom
        self.nTweezers = nTweezers
        if positions is not None:
            self.positions = positions
            self.nTweezers = len(positions)
        else:
            self.positions = None
            if self.nTweezers is None:
                raise ValueError('Both positions and nTweezers are not given! Cannot further process!')
            self.nTweezers = nTweezers

    def findPositions(self, plot, tranpose=True):
        self.positions = findTweezerPositions_v2(self.imageStack, self.nTweezers, self.rAtom, plot=plot,
                                                 transpose=tranpose)
        return self.positions

    def calculateWeights(self, gaussianRadius=2, bgExcludeRegion=None):
        self.pixelWeight = []
        summedImage = self.summedImage
        nTweezers = self.nTweezers
        pixelPosition = []
        for k in range(nTweezers):
            dummy = []
            for i in range(-self.rAtom, self.rAtom + 1):
                for j in range(-self.rAtom, self.rAtom + 1):
                    dummy.append([self.positions[k][0] + i, self.positions[k][1] + j])
                    # All the pixels related to the k-th tweezer.
            pixelPosition.append(dummy)
        pixelPosition = np.asarray(pixelPosition)  # [nTweezers, (2*rAtom+1)^2]
        self.pixelPosition = pixelPosition

        if bgExcludeRegion is not None:  # Get the average counts of the background.
            mask = np.ones(summedImage.shape, dtype=bool)
            bgr = bgExcludeRegion
            mask[bgr[0][0]:bgr[1][0], bgr[0][1]:bgr[1][1]] = False
            self.bkMask = mask
            bkGround = summedImage[mask].mean()
        else:
            self.bkMask = np.zeros_like(summedImage, dtype=bool)
            bkGround = 0

        for i in range(nTweezers):
            for j in range((2 * self.rAtom + 1) ** 2):
                x, y = pixelPosition[i][j][0] - self.positions[i][0], pixelPosition[i][j][1] - self.positions[i][1]
                gradPixel = np.exp(-(x ** 2 + y ** 2) / (gaussianRadius ** 2))
                self.pixelWeight.append([pixelPosition[i][j], gradPixel])
        return None

    def applyWeights(self, weights=False, transpose=True, fitParams=None):
        '''
        Edited by Pai 05252023
        Added additional return variables r2 and warningFlag. r2 is the R^2 for all fits. warningFlag list of strings with length equal to number of tweezers. The string is empty if no warning, 'ThresholdLow' if threshold < x_d + 1.5 sigma_d, 'R2Low' if r2 < 0.99.
        '''
        def doubleGaussian(x, a1, a2, x1, x2, w1, w2):
            return a1 * np.exp(-(x - x1) ** 2 / (2 * w1 ** 2)) + a2 * np.exp(-(x - x2) ** 2 / (2 * w2 ** 2))
        def gauss(x,a,x0,w):
            return a*np.exp(-(x-x0)**2/2/w**2)
        def fitDoubleGaussian(x, y, p0=5):
            '''
            Added by Pai 05252023
            Fit a histogram curve to the sum of two gaussian functions.
            The function works by first fit the dark peak (higher one) to a single gaussian. Use the peak position and std of the data as the initial guess. Then subtract the dark peak from the histogram and use the same method to fit the bright curve. Then we use the fitted results of the two peaks as the initial guess for a final double gaussian fit.
            Input: x, y are the data to fit. p0 is the initial guess of the width of the dark peak.
            Return: 
                fit: optimal parameters
                r_squared: R^2
                thresh: threshold optained from MEL
                fit1, fit2: optimal parameter for the single gaussian fits
            
            '''
            try:
                # we fit only the [x < x_d + sigma_d] region
                window = x < x[np.argmax(y)]+1*p0
                # estimate the std = \sqrt{\int p(x) x**2 - (\int p(x) x)**2}
                s01 = np.sqrt(np.sum(y[window]*(x[window]**2))/np.sum(y[window])-(np.sum(y[window]*x[window])/np.sum(y[window]))**2)
                fit1, _ = curve_fit(gauss, x[window], y[window], p0=(np.max(y[window]), x[window][np.argmax(y[window])], s01), 
                                    bounds=((0,np.min(x[window]), 0), (2*np.max(y[window]),np.max(x[window]),np.inf)))
                
                p0 = fit1
                # subtract the dark peak and fit only [x > x_d + 2 sigma_d] region
                res = np.clip(y-gauss(x,*fit1), a_min=0, a_max=np.inf)
                window = x > p0[1]+2*p0[2]
                s02 = np.sqrt(np.sum(res[window]*(x[window]**2))/np.sum(res[window])-(np.sum(res[window]*x[window])/np.sum(res[window]))**2)
                fit2,_ = curve_fit(gauss, x[window], res[window], p0=(np.max(res[window]), x[window][np.argmax(res[window])], s02), 
                                    bounds=((0,np.min(x[window]), 0), (2*np.max(res[window]),np.max(x[window]),np.inf)))
                # use the above fitted results as the initial guess for a double gaussian fit
                p0=np.array([[fit1[i], fit2[i]] for i in range(3)]).flatten()
                fit, _ = curve_fit(doubleGaussian, x, y, p0=p0, 
                                   bounds=((0,0,np.min(x), np.min(x),0,0), (2*np.max(y),2*np.max(y),np.max(x), np.max(x),np.inf,np.inf)))
                data_fitted = doubleGaussian(x, *fit)
                residuals = y - data_fitted
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y-np.mean(y))**2)
                r_squared = 1 - (ss_res / ss_tot)
                # make sure x_1 < x_2
                if fit[3] < fit[2]:
                    fit=fit[[1,0,3,2,5,4]]
                fit[4:] = np.abs(fit[4:])
                x1, x2, w1, w2 = fit[2:]

                # MLE discrimination of two gaussian
                Delta=2*(np.log(w1)-np.log(w2))*(w1-w2)*(w1+w2)+(x1-x2)**2
                thresh=(w1**2*x2-w2**2*x1-w1*w2*np.sqrt(Delta))/(w1**2-w2**2)
            except:
                # if fitting failed, just return NaN
                fit1 = [np.nan]*3
                fit2 = [np.nan]*3
                fit = [np.nan]*6
                r_squared = np.nan
                thresh=1e4
            return fit, r_squared, thresh, fit1, fit2

        files = self.imageStack
        nImages = self.nImages
        numPixels = len(self.pixelWeight)
        mask = np.ones((np.shape(files)[1], np.shape(files)[2]))
        weightedImages = []
        self.bk = []
        if weights:
            for i in range(numPixels):
                mask[int(self.pixelWeight[i][0][0]), int(self.pixelWeight[i][0][1])] = self.pixelWeight[i][1]

            if transpose:
                mask_for_plot = mask.T
            else:
                mask_for_plot = mask
            plt.figure(figsize=(3, 3 * np.shape(mask_for_plot)[0] / np.shape(mask_for_plot)[0]))
            plt.imshow(mask_for_plot)
            plt.show()

            for j in range(nImages):
                self.bk.append(files[j][self.bkMask].mean() * (2 * self.rAtom + 1) ** 2)
                weightedImages.append((files[j]) * mask)
        else:
            for j in range(nImages):
                self.bk.append(files[j][self.bkMask].mean() * (2 * self.rAtom + 1) ** 2)
                weightedImages.append((files[j]))
        weightedImages = np.array(weightedImages)
        self.weightedImages = weightedImages
        self.mask = mask
        counts = getCounts(weightedImages, self.positions, rAtom=self.rAtom)
        fits = []
        r2 = []
        thresholds = []
        warningFlag = []

        n_try = 1
        for n_try in range(5, 0, -1):  # Reshape the nTweezer plots into a more organized shape = (Lx * Ly).
            if np.mod(self.nTweezers, n_try) == 0:
                break
        num_plot_x = n_try
        num_plot_y = self.nTweezers // n_try
        fig, ax = plt.subplots(num_plot_y, num_plot_x, figsize=(3 * num_plot_x, 3 * num_plot_y))
        if self.nTweezers > 1:
            for i in range(self.nTweezers):
                ax_idx_0, ax_idx_1 = np.unravel_index(i, (num_plot_y, num_plot_x))
                y, x, _ = ax[ax_idx_0, ax_idx_1].hist(counts[:, i], bins=50, label='{}:{}'.format(i, self.positions[i]),
                                                      alpha=0.7, color='tab:blue')
                ax[ax_idx_0, ax_idx_1].legend(loc='upper right')
                ax[ax_idx_0, ax_idx_1].set_ylim([0, nImages / 10])
                if fitParams is not None:
                    xc = np.array([0.5 * (x[i] + x[i + 1]) for i in range(len(x) - 1)])
                    xx = np.linspace(np.amin(xc), np.amax(xc), 1000)
                    
                    fit,r_squared,thresh,fit1,fit2 = fitDoubleGaussian(xc, y, p0=fitParams[i])
                    x1, x2, w1, w2 = fit[2:]
                    flag = ''
                    if (thresh-fit[2])/fit[4]<1.5:
                        print('Threshold too low at site {:d}'.format(i))
                        flag='ThresholdLow '
                    if r_squared < 0.99:
                        print('R2 = {:.3f} at site {:d}'.format(r_squared, i))
                        flag+='R2Low '
                        
                    # change the title color to red if there is a warning
                    if len(flag) > 0:
                        titleColor='red'
                    else:
                        titleColor='black'
                    thresholds.append(thresh)
                    if not(np.isnan(r_squared)):
                        
                        ax[ax_idx_0, ax_idx_1].plot(xx, gauss(xx, *fit1))
                        ax[ax_idx_0, ax_idx_1].plot(xx, gauss(xx, *fit2))
                        ax[ax_idx_0, ax_idx_1].plot(xx, doubleGaussian(xx, *fit), color='black')
                        
                        ax[ax_idx_0, ax_idx_1].set_title(
                        '$a_d$ = {:.2f}, $x_d$ = {:.2f}, $\sigma_d$ = {:.2f}, t={:.2f} \n$a_b$ = {:.2f}, $x_b$ = {:.2f}, $\sigma_b$ = {:.2f}, r2 = {:.3f}'.format(fit[0],
                            x1, np.abs(w1), thresh, fit[1], x2, np.abs(w2), r_squared), fontsize=7,color=titleColor)
                        ax[ax_idx_0, ax_idx_1].axvline(x=thresh, linestyle='--', color='red')
                    

                    fits.append(fit)
                    r2.append(r_squared)
                    warningFlag.append(flag)
        elif self.nTweezers == 1:
            i = 0
            y, x, _ = ax.hist(counts[:, i], bins=50, label='position={}'.format(self.positions[i]))
            ax.legend(loc='upper right')
            ax.set_ylim([0, nImages / 10])
            if fitParams is not None:
                xc = np.array([0.5 * (x[i] + x[i + 1]) for i in range(len(x) - 1)])
                xx = np.linspace(np.amin(xc), np.amax(xc), 1000)
                fit,r_squared,thresh,fit1,fit2 = fitDoubleGaussian(xc, y, p0=fitParams[i])
                x1, x2, w1, w2 = fit[2:]
                flag = ''
                if (thresh-fit[2])/fit[4]<1.5:
                    print('Threshold too low at site {:d}'.format(i))
                    flag='ThresholdLow '
                if r_squared < 0.99:
                    print('R2 = {:.3f} at site {:d}'.format(r_squared, i))
                    flag+='R2Low '
                thresholds.append(thresh)
                if not(np.isnan(r_squared)):
                    ax.plot(xx, doubleGaussian(xx, *fit), color='r')
                    ax.axvline(x=thresh, linestyle='--')
                fits.append(fit)
                r2.append(r_squared)
                warningFlag.append(flag)
        plt.tight_layout()
        plt.show()

        thresholds = np.array(thresholds)
        self.thresholds = thresholds

        if fitParams is not None:
            return thresholds.flatten(), mask, fits, r2, warningFlag
        else:
            return 0, mask, fits, r2
    
    def applyWeightsOld(self, weights=False, transpose=True, fitParams=None):
        def doubleGaussian(x, a1, a2, x1, x2, w1, w2):
            return a1 * np.exp(-(x - x1) ** 2 / (2 * w1 ** 2)) + a2 * np.exp(-(x - x2) ** 2 / (2 * w2 ** 2))

        files = self.imageStack
        nImages = self.nImages
        numPixels = np.shape(self.pixelWeight)[0]
        mask = np.ones((np.shape(files)[1], np.shape(files)[2]))
        weightedImages = []
        self.bk = []
        if weights:
            for i in range(numPixels):
                mask[self.pixelWeight[i][0][0], self.pixelWeight[i][0][1]] = self.pixelWeight[i][1]

            if transpose:
                mask_for_plot = mask.T
            else:
                mask_for_plot = mask
            plt.figure(figsize=(3, 3 * np.shape(mask_for_plot)[0] / np.shape(mask_for_plot)[0]))
            plt.imshow(mask_for_plot)
            plt.show()

            for j in range(nImages):
                self.bk.append(files[j][self.bkMask].mean() * (2 * self.rAtom + 1) ** 2)
                weightedImages.append((files[j]) * mask)
        else:
            for j in range(nImages):
                self.bk.append(files[j][self.bkMask].mean() * (2 * self.rAtom + 1) ** 2)
                weightedImages.append((files[j]))
        weightedImages = np.array(weightedImages)
        self.weightedImages = weightedImages
        self.mask = mask
        counts = getCounts(weightedImages, self.positions, rAtom=self.rAtom)
        fits = []
        thresholds = []

        n_try = 1
        for n_try in range(5, 0, -1):  # Reshape the nTweezer plots into a more organized shape = (Lx * Ly).
            if np.mod(self.nTweezers, n_try) == 0:
                break
        num_plot_x = n_try
        num_plot_y = self.nTweezers // n_try
        fig, ax = plt.subplots(num_plot_y, num_plot_x, figsize=(3 * num_plot_x, 3 * num_plot_y))
        if self.nTweezers > 1:
            for i in range(self.nTweezers):
                ax_idx_0, ax_idx_1 = np.unravel_index(i, (num_plot_y, num_plot_x))
                y, x, _ = ax[ax_idx_0, ax_idx_1].hist(counts[:, i], bins=50, label='{}:{}'.format(i, self.positions[i]),
                                                      alpha=0.7, color='tab:blue')
                ax[ax_idx_0, ax_idx_1].legend(loc='upper right')
                ax[ax_idx_0, ax_idx_1].set_ylim([0, nImages / 10])
                if fitParams is not None:
                    xc = np.array([0.5 * (x[i] + x[i + 1]) for i in range(len(x) - 1)])
                    xx = np.linspace(np.amin(xc), np.amax(xc), 1000)
                    fit, _ = curve_fit(doubleGaussian, xc, y, p0=fitParams[i])
                    x1, x2, w1, w2 = fit[2:]
                    x1, x2 = np.abs(x1), np.abs(x2)
                    f1, f2 = np.sqrt(np.abs(w1) / 2), np.sqrt(np.abs(w2) / 2)
                    thresh = (x2 / f2 + x1 / f1) / (1 / f1 + 1 / f2)
                    thresholds.append(thresh)
                    ax[ax_idx_0, ax_idx_1].plot(xx, doubleGaussian(xx, *fit), color='black')
                    ax[ax_idx_0, ax_idx_1].axvline(x=thresh, linestyle='--', color='red')
                    ax[ax_idx_0, ax_idx_1].set_title(
                        '$x_d$ = {:.2f}, $\sigma_d$ = {:.2f} \n$x_b$ = {:.2f}, $\sigma_b$ = {:.2f}'.format(
                            x1, np.abs(w1), x2, np.abs(w2)))

                    fits.append(fit)
        elif self.nTweezers == 1:
            i = 0
            y, x, _ = ax.hist(counts[:, i], bins=50, label='position={}'.format(self.positions[i]))
            ax.legend(loc='upper right')
            ax.set_ylim([0, nImages / 10])
            if fitParams is not None:
                xc = np.array([0.5 * (x[i] + x[i + 1]) for i in range(len(x) - 1)])
                xx = np.linspace(np.amin(xc), np.amax(xc), 1000)
                fit, _ = curve_fit(doubleGaussian, xc, y, p0=fitParams[i])
                x1, x2, w1, w2 = fit[2:]
                x1, x2 = np.abs(x1), np.abs(x2)
                f1, f2 = np.sqrt(np.abs(w1) / 2), np.sqrt(np.abs(w2) / 2)
                thresh = (x2 / f2 + x1 / f1) / (1 / f1 + 1 / f2)
                thresholds.append(thresh)
                ax.plot(xx, doubleGaussian(xx, *fit), color='r')
                ax.axvline(x=thresh, linestyle='--')
                fits.append(fit)
        plt.tight_layout()
        plt.show()

        thresholds = np.array(thresholds)
        self.thresholds = thresholds

        if fitParams is not None:
            return thresholds.flatten(), mask, fits
        else:
            return 0, mask, fits

    def fidelityAnalysis(self, plot=True):
        occ = np.array([getOccupancy(
            scaleImageStack(f.load(), cameraType=self.cameraType), self.thresholds, self.positions, self.rAtom,
            weights=self.mask) for f in self.files])  # [nFiles, nLoops * ImagesPerLoop, nTweezers]
        s = occ.shape
        occ2 = np.reshape(occ, (s[0], self.nLoops, -1, s[-1]))  # [nFiles, nLoops, ImagesPerLoop, nTweezers]
        if self.exclude1st:
            occ2 = occ2[:, :, 1:, :]

        diffs = -np.diff(occ2, axis=2)
        # bright -> dark probability:
        nbd = np.count_nonzero(diffs == 1, axis=2).sum(axis=(0, 1))
        nbdChances = np.count_nonzero(occ2[:, :, :-1, :] == 1, axis=2).sum(axis=(0, 1))
        pbd = nbd / nbdChances

        # dark -> bright probability:
        ndb = np.count_nonzero(diffs == -1, axis=2).sum(axis=(0, 1))
        ndbChances = np.count_nonzero(occ2[:, :, :-1, :] == 0, axis=2).sum(axis=(0, 1))
        pdb = ndb / ndbChances

        probs = {'darkToBright': pdb, 'brightToDark': pbd}

        if plot:
            plt.figure(figsize=(5, 5 / 1.5))
            plt.plot(probs['darkToBright'], 'o', color='black', label=r'$d \rightarrow b$')
            plt.axhline(y=probs['darkToBright'].mean(), color='grey', linestyle='dashed')
            plt.plot(probs['brightToDark'], 'o', color='tab:blue', label=r'$b \rightarrow d$')
            plt.axhline(y=probs['brightToDark'].mean(), color='darkblue', linestyle='dashed')
            plt.legend()
            plt.ylabel('Probability')
            plt.xlabel('Sites')
            plt.tight_layout()
            probs['brightToDark'][np.isnan(probs['brightToDark'])] = 0  # In case we are having a hard time loading
            # atoms.
            plt.title('bright to dark Prob averaged: ' + str(
                np.round(np.nanmean(probs['brightToDark']), 5)) + '\n dark to bright Prob averaged: ' + str(
                np.round(np.nanmean(probs['darkToBright']), 5)))
            plt.ylim(-0.01, 0.1)
            plt.show()
        return probs


class weighted_histogram_399:

    def __init__(self, files, answers, nTweezers=None, positions=None, activeIdx=None, rAtom=2, bgExcludeRegion=None):
        """
        answers: should be of shape [nFiles * nLoops, nTweezers] or [nFiles, nLoops, nTweezers].
        """
        self.cameraType = 'nuvu'
        try:
            iter(files)
        except:  # single file
            self.imageStack = scaleImageStack(files.load(), cameraType=self.cameraType)
        else:  # multiple files
            dum = np.array([scaleImageStack(f.load(), cameraType=self.cameraType) for f in files])
            self.imageStack = np.reshape(dum, (-1, dum.shape[-2], dum.shape[-1]))  # [nFiles * nLoops, length_x,
            # length_y]
        self.files = files
        self.nImages = np.shape(self.imageStack)[0]
        self.summedImage = self.imageStack.sum(axis=0)
        self.pixelWeight = None
        self.thresholds = None
        self.rAtom = rAtom
        self.bgExcludeRegion = bgExcludeRegion
        if positions is not None:
            if activeIdx is None:
                self.positions = positions
            else:
                self.positions = positions[activeIdx]
            self.nTweezers = len(self.positions)
        else:
            self.positions = None
            if nTweezers is None:
                raise ValueError('Both positions and nTweezers are not given! Cannot further process!')
            self.nTweezers = nTweezers

        if answers is not None:
            self.answers = np.reshape(answers, (-1, self.nTweezers))
            if np.shape(self.answers)[0] != self.nImages:
                raise ValueError('The number of answers ({:.0f}) is not the same with the number of images ({:.0f})!'
                                 .format(np.shape(self.answers)[0], self.nImages))

    def findPositions(self, plot, transpose=True):
        self.positions = findTweezerPositions_v2(self.imageStack, self.nTweezers, self.rAtom, plot=plot,
                                                 transpose=transpose)
        return self.positions
    
    def plotHistogram(self, bin_step=0.5, thresh_step=0.01):
        cnts_bgSub = getBkgSubCounts(self.imageStack, self.positions, rAtom=self.rAtom,
                                     bgExcludeRegion=self.bgExcludeRegion)
        cnts_bgSub = cnts_bgSub.flatten()
        adc_cnts_list = np.arange(np.amin(cnts_bgSub), np.amax(cnts_bgSub) + bin_step, bin_step)
        
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5 / 1.5))
        nImg, bins, _ = ax.hist(cnts_bgSub, bins=adc_cnts_list, alpha=0.5, color='tab:blue')
        ax.set_xlabel('Counts')
        ax.set_ylabel('#Images')
        plt.show()
        return nImg, bins
        
    def findThresholdsAndFidelity(self, single_thresh=True, bin_step=0.5, thresh_step=0.01, plot=True):
        cnts_bgSub = getBkgSubCounts(self.imageStack, self.positions, rAtom=self.rAtom,
                                     bgExcludeRegion=self.bgExcludeRegion)

        cnts_bgSub_0 = cnts_bgSub * (1 - self.answers)  # [..., #Tweezers]
        cnts_bgSub_1 = cnts_bgSub * self.answers  # [..., #Tweezers]

        if single_thresh:
            cnts_bgSub_0 = cnts_bgSub_0.flatten()
            cnts_bgSub_1 = cnts_bgSub_1.flatten()
            cnts_bgSub_0 = cnts_bgSub_0[cnts_bgSub_0 != 0]
            cnts_bgSub_1 = cnts_bgSub_1[cnts_bgSub_1 != 0]
            adc_cnts_list = np.arange(np.amin(cnts_bgSub), np.amax(cnts_bgSub) + bin_step, bin_step)
            thresholdList = np.arange(np.amin(cnts_bgSub), np.amax(cnts_bgSub), thresh_step)
            false_positive = np.array(
                list(map(lambda x: np.where(cnts_bgSub_0 > x)[0].shape[0], thresholdList)))
            false_negative = np.array(
                list(map(lambda x: np.where(cnts_bgSub_1 < x)[0].shape[0], thresholdList)))
            index = np.argmin(np.abs(false_positive / len(cnts_bgSub_0) - false_negative / len(cnts_bgSub_1)))
            thresh = thresholdList[index]
            fidelity = 1 - (false_positive[index] / len(cnts_bgSub_0) + false_negative[index] / len(cnts_bgSub_1)) / 2
            if plot:
                fig, ax = plt.subplots(1, 2, figsize=(7, 3.5 / 1.5))
                nImg0, bin0, _ = ax[0].hist(cnts_bgSub_0, bins=adc_cnts_list, alpha=0.5,
                                            color='tab:blue', label='No atoms')
                nImg1, bin1, _ = ax[0].hist(cnts_bgSub_1, bins=adc_cnts_list, alpha=0.5,
                                            color='tab:green', label='Atoms')
                x0 = bin0[1:] + bin0[:-1]
                x1 = bin1[1:] + bin1[:-1]
                ax[0].set_xlabel('Counts')
                ax[0].set_ylabel('#Images')
                ax[0].legend()
                ax[0].set_title('Avg counts of an atom: {:.1f}'.format(np.mean(cnts_bgSub_1)))
                ax[1].plot(thresholdList, false_positive / len(cnts_bgSub_0), label='False positive rate')
                ax[1].plot(thresholdList, false_negative / len(cnts_bgSub_1), label='False negative rate')
                ax[1].set_yscale('log')
                ax[1].axvline(x=thresh, linestyle='--', color='red', label='threshold')
                ax[1].set_ylim(5e-4, 1)
                ax[1].legend()
                ax[1].set_title('Threshold: {:.4f}; Fidelity: {:.4f}'.format(thresh, fidelity))
                ax[1].set_xlabel('Counts')
                plt.show()
            return thresh

        else:  # When one wants a different threshold for different tweezer sites.
            thresholds = []
            if plot:
                fig, ax = plt.subplots(self.nTweezers, 2, figsize=(7, 3.5 / 1.5 * self.nTweezers))
            for i in range(self.nTweezers):
                cnts_bgSub_0_site = cnts_bgSub_0[:, i].flatten()
                cnts_bgSub_1_site = cnts_bgSub_1[:, i].flatten()
                cnts_bgSub_0_site = cnts_bgSub_0_site[cnts_bgSub_0_site != 0]
                cnts_bgSub_1_site = cnts_bgSub_1_site[cnts_bgSub_1_site != 0]
                adc_cnts_list = np.arange(np.amin(cnts_bgSub), np.amax(cnts_bgSub) + bin_step, bin_step)
                thresholdList = np.arange(np.amin(cnts_bgSub), np.amax(cnts_bgSub), thresh_step)
                false_positive = np.array(
                    list(map(lambda x: np.where(cnts_bgSub_0_site > x)[0].shape[0], thresholdList)))
                false_negative = np.array(
                    list(map(lambda x: np.where(cnts_bgSub_1_site < x)[0].shape[0], thresholdList)))
                index = np.argmin(np.abs(false_positive / len(cnts_bgSub_0_site) -
                                         false_negative / len(cnts_bgSub_1_site)))
                thresh = thresholdList[index]
                fidelity = 1 - (false_positive[index] / len(cnts_bgSub_0_site) +
                            false_negative[index] / len(cnts_bgSub_1_site)) / 2
                if plot:
                    nImg0, bin0, _ = ax[i][0].hist(cnts_bgSub_0_site, bins=adc_cnts_list, alpha=0.5,
                                                color='tab:blue', label='No atoms')
                    nImg1, bin1, _ = ax[i][0].hist(cnts_bgSub_1_site, bins=adc_cnts_list, alpha=0.5,
                                                color='tab:green', label='Atoms')
                    x0 = bin0[1:] + bin0[:-1]
                    x1 = bin1[1:] + bin1[:-1]
                    ax[i][0].set_xlabel('Counts')
                    ax[i][0].set_ylabel('#Images')
                    ax[i][0].legend(title='Site #{}: {}'.format(i, self.positions[i]))
                    ax[i][0].set_title('Avg counts of an atom: {:.1f}'.format(np.mean(cnts_bgSub_1_site)))
                    ax[i][1].plot(thresholdList, false_positive / len(cnts_bgSub_0_site), label='False positive rate')
                    ax[i][1].plot(thresholdList, false_negative / len(cnts_bgSub_1_site), label='False negative rate')
                    ax[i][1].set_yscale('log')
                    ax[i][1].axvline(x=thresh, linestyle='--', color='red', label='threshold')
                    ax[i][1].set_ylim(5e-4, 1)
                    ax[i][1].legend()
                    ax[i][1].set_title('Threshold: {:.4f}; Fidelity: {:.4f}'.format(thresh, fidelity))
                    ax[i][1].set_xlabel('Counts')
                thresholds.append(thresh)
            if plot:
                plt.show()
            return thresholds
