/*
 * Vectors docs — dark/light theme toggle + navbar scroll state.
 * Ports the MFCQI-java behaviour: the theme lives in a `data-theme` attribute on
 * <html>, persisted to localStorage. The pre-paint initialisation (which reads the
 * stored value / system preference before first render to avoid a flash) lives inline
 * in head-meta.hbs; this deferred script only wires up the interactive bits once the
 * DOM exists.
 */
(function () {
  'use strict';

  var STORAGE_KEY = 'vectors-theme';
  var root = document.documentElement;

  // Theme toggle button (sun/moon in the header).
  var toggle = document.getElementById('vc-theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
      root.setAttribute('data-theme', next);
      try {
        localStorage.setItem(STORAGE_KEY, next);
      } catch (e) {
        /* storage unavailable — non-fatal */
      }
    });
  }

  // Follow the OS theme when the user hasn't made an explicit choice.
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    var onSchemeChange = function (e) {
      var stored;
      try {
        stored = localStorage.getItem(STORAGE_KEY);
      } catch (err) {
        stored = null;
      }
      if (!stored) root.setAttribute('data-theme', e.matches ? 'dark' : 'light');
    };
    if (mq.addEventListener) mq.addEventListener('change', onSchemeChange);
    else if (mq.addListener) mq.addListener(onSchemeChange);
  }

  // Frosted-navbar border appears once the page is scrolled (MFCQI `.scrolled`).
  var navbar = document.querySelector('.navbar');
  if (navbar) {
    var onScroll = function () {
      navbar.classList.toggle('is-scrolled', window.pageYOffset > 40);
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
  }
})();
