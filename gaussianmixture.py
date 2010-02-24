#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-


__author__ = 'Justin Bayer, bayer.justin@googlemail.com'


import optparse
import sys

import scipy
import scipy.linalg
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab

import theano
import theano.tensor as T
from theanoext import inv as T_inv, det as T_det


def make_optparse():
  parser = optparse.OptionParser()
  parser.add_option('--degree', type='int', dest='amount_means',
                    help='adjust the amount of means of the mixture')
  parser.add_option('--epochs', type='int', dest='epochs', default=100,
                    help='specify the number of epochs to train')
  parser.add_option('--xfilename', type='str', dest='xfilename',
                    help='give the path to the data file containing x values')
  parser.add_option('--yfilename', type='str', dest='yfilename',
                    help='give the path to the data file containing y values')
  parser.add_option('--plot', action='store_true', dest='plot',
                    help="""plot the data and the mixture - only for 1D and 2D
                    data""")

  return parser


class MultivariateGaussianDensity(object):

  mean = T.vector('mean')
  covariance = T.matrix('covariance')
  dim = T.scalar('dim')

  norm_expr = ((2 * scipy.pi) ** (mean.shape[0] / 2) *
               T.sqrt(abs(T_det(covariance))))
  _norm = theano.function([mean, covariance, dim], norm_expr)

  inpt = T.vector('inpt')

  precision = T_inv(covariance)
  centered = inpt - mean
  exponent_expr = T.exp(-0.5 * T.dot(T.dot(centered, precision), centered))
  _exponent = theano.function([inpt, mean, covariance], exponent_expr)

  def __init__(self, mean, cov):
    self.mean = scipy.asarray(mean)
    self.cov = scipy.asarray(cov)
    self.dim = self.mean.shape[0]

  def pdf(self, x):
    res = self._exponent(x, self.mean, self.cov)
    res /= self._norm(self.mean, self.cov)
    return ret

  def multpdf(self, xs):
    xs = scipy.asarray(list(xs))
    res = scipy.empty(xs.shape[0])
    norm = self._norm(self.mean, self.cov, self.dim)
    for i, x in enumerate(xs):
      res[i] = self._exponent(x, self.mean, self.cov)
    return res / norm


class GaussianMixture(object):

  def __init__(self, mixcoeffs, means, covs):
    self.mixcoeffs = scipy.asarray(mixcoeffs)
    self.means = scipy.asarray(means)
    self.covs = scipy.asarray(covs)

    self.degree = self.mixcoeffs.shape[0]
    self.dim = self.covs.shape[1]

  @classmethod
  def randomized(cls, degree, dim, scale):
    mixcoeffs = scipy.random.random(degree)
    mixcoeffs /= mixcoeffs.sum()

    means = scipy.random.standard_normal((degree, dim)) * scale

    randomdata = (scipy.random.standard_normal((dim, 10)) * scale
                     for _ in xrange(degree))
    covs = [scipy.cov(i) for i in randomdata]
    return cls(mixcoeffs, means, covs)

  def fit(self, data, epochs=100):
    data = scipy.asarray(data)
    for i in range(epochs):
      self.fit_once(data)

  def _responsibilities(self, data):
    """Return a 2D-array where the item [n, k] contains the probabilities that 
    point n belongs to mode k."""
    resps = scipy.empty((len(data), self.degree))
    for i in range(self.degree):
      mvg = MultivariateGaussianDensity(self.means[i], self.covs[i])
      p = self.mixcoeffs[i] * mvg.multpdf(data)
      p.shape = p.shape[0],
      resps[:, i] = p

    # OPT: without transpose?!
    resps = resps.T
    resps /= resps.sum(axis=0)
    resps = resps.T

    return resps

  def _point_expectations(self, data, resps):
    """Return the expected points per mode.

    Bishop (9.27)"""
    point_exp = resps.sum(axis=0)
    return point_exp

  def _means(self, data, resps, point_exp):
    """Return a 2D-array where the i'th row contains the mean for the i'th
    mode.

    Bishop (9.24)"""
    means = scipy.empty((self.degree, self.dim))
    for i in range(self.degree):
      this_resps = resps[:, i].reshape((resps.shape[0], 1))
      means[i] = (this_resps * data).sum(axis=0) / point_exp[i]
    return means

  def _covs(self, data, resps, point_exp, means):
    """Return a 3D-array where the i'th row contains the covariance matrix for
    the i'th mode.

    Bishop (9.25)"""
    covs = scipy.empty((self.degree, self.dim, self.dim))
    for i in range(self.degree):
      centered = data - means[i]
      this_resps = resps[:, i].reshape((resps.shape[0], 1))
      weighted = (centered * scipy.sqrt(this_resps)).T
      # Using the scipy cov here somehow results in non-positive semidefinite
      # covariance matrices.
      this_cov = scipy.dot(weighted, weighted.T) / point_exp[i]
      covs[i, :, :] = this_cov
    return covs

  def _mixcoeffs(self, data, point_exp):
    """Calculate the new mixing coefficients.

    Bishop (9.26)"""
    return point_exp / data.shape[0]

  def fit_once(self, data):
    # Calculate the probability that a point "belongs" to a given mode.
    # Bishop (9.23).
    self.resps =  self._responsibilities(data)
    point_exp = self._point_expectations(data, self.resps)
    self.means = self._means(data, self.resps, point_exp)
    self.covs = self._covs(data, self.resps, point_exp, self.means)
    self.mixcoeffs = self._mixcoeffs(data, point_exp)


def load_data(xfile, yfile):
  loadfile = lambda fn: [float(i.strip()) for i in open(fn).xreadlines() 
                         if i.strip()]
  xs = loadfile(xfile)
  ys = loadfile(yfile)
  return xs, ys


def plot_mixture(fig, mixture, data):
  plt.subplot(fig)

  # Plot data.
  plt.scatter(data[0], data[1], color='r', marker='+', alpha=0.5)

  delta = 1.0
  coords = []
  xs = scipy.arange(-30, 30, delta)
  ys = scipy.arange(-30, 30, delta)
  for x in xs:
    for y in ys:
      coords.append((x, y))

  coords = scipy.asarray(coords)

  Z = None
  for i in range(mixture.degree):
    #z = mixture.mixcoeffs[i] * mgd(coords, mixture.means[i], mixture.covs[i])

    gaussian = MultivariateGaussianDensity(mixture.means[i], mixture.covs[i])
    z = mixture.mixcoeffs[i] * gaussian.multpdf(coords)

    if Z is None:
      Z = z
    else:
      Z += z

  X, Y = scipy.meshgrid(xs, ys)
  Z.shape = X.shape

  CS = plt.contour(X, Y, Z, 30)
  plt.plot(mixture.means[:, 1], mixture.means[:, 0], 'o', color='green')
  print mixture.means


def main():
  options, args = make_optparse().parse_args()

  data = load_data(options.xfilename, options.yfilename)

  mixture = GaussianMixture.randomized(4, 2, 30)

  plt.ion()
  for i in range(10):
    mixture.fit(zip(*data), epochs=options.epochs / 10)
    plot_mixture(111, mixture, data)
    raw_input("Hit return to continue fitting...")
    plt.clf()

  print mixture.mixcoeffs
  print "-" * 80
  print mixture.means
  print "-" * 80
  print mixture.covs
  print "-" * 80

  return 0


if __name__ == '__main__':
  sys.exit(main())
