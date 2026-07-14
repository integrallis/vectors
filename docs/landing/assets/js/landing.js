// Vectors landing — theme switch, nav state, scroll reveal, copy-to-clipboard.
// Adapted from the MFCQI landing; the theme is stored under the same key the Antora
// docs use ('vectors-theme') so a choice made here carries over to the docs and back.

document.addEventListener('DOMContentLoaded', () => {
  // ---- Java syntax highlighting (highlight.js, bundled locally) ---------
  if (window.hljs) {
    const paint = window.hljs.highlightElement || window.hljs.highlightBlock;
    document.querySelectorAll('.code-panel pre code').forEach((el) => {
      try {
        paint.call(window.hljs, el);
      } catch (e) {
        /* leave the block un-highlighted rather than break the page */
      }
    });
  }

  // ---- Theme toggle (persisted; defaults to system preference) ----------
  const root = document.documentElement;
  const toggle = document.getElementById('theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', () => {
      const next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
      root.setAttribute('data-theme', next);
      try {
        localStorage.setItem('vectors-theme', next);
      } catch (e) {
        /* ignore */
      }
    });
  }

  // ---- Nav scroll state -------------------------------------------------
  const nav = document.querySelector('.nav');
  const onScroll = () => nav.classList.toggle('scrolled', window.pageYOffset > 40);
  onScroll();
  window.addEventListener('scroll', onScroll, { passive: true });

  // ---- Smooth anchor scroll --------------------------------------------
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener('click', function (e) {
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        e.preventDefault();
        const top = target.getBoundingClientRect().top + window.pageYOffset - 76;
        window.scrollTo({ top, behavior: 'smooth' });
      }
    });
  });

  // ---- Scroll reveal ----------------------------------------------------
  const revealEls = document.querySelectorAll('.reveal');
  const reveal = () => {
    const h = window.innerHeight;
    revealEls.forEach((el) => {
      if (el.getBoundingClientRect().top < h - 110) el.classList.add('visible');
    });
  };
  reveal();
  let ticking = false;
  window.addEventListener(
    'scroll',
    () => {
      if (!ticking) {
        window.requestAnimationFrame(() => {
          reveal();
          ticking = false;
        });
        ticking = true;
      }
    },
    { passive: true }
  );

  // ---- Code-showcase tabs (Java / Spring AI / LangChain4j) -------------
  const codeTabs = document.querySelectorAll('.code-tab');
  const codePanels = document.querySelectorAll('.code-panel');
  codeTabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      const key = tab.dataset.code;
      codeTabs.forEach((t) => t.classList.toggle('active', t === tab));
      codePanels.forEach((p) => p.classList.toggle('active', p.dataset.code === key));
    });
  });

  // ---- Dependency coordinates (build-system tabs) ----------------------
  const COORDS = {
    'gradle-kts': {
      copy: 'implementation("com.integrallis:vectors:0.1.0")',
      show: 'implementation("com.integrallis:vectors:0.1.0")',
    },
    'gradle-groovy': {
      copy: "implementation 'com.integrallis:vectors:0.1.0'",
      show: "implementation 'com.integrallis:vectors:0.1.0'",
    },
    maven: {
      copy: '<dependency>\n  <groupId>com.integrallis</groupId>\n  <artifactId>vectors</artifactId>\n  <version>0.1.0</version>\n</dependency>',
      show: '<dependency> com.integrallis:vectors:0.1.0 </dependency>',
    },
    sbt: {
      copy: 'libraryDependencies += "com.integrallis" % "vectors" % "0.1.0"',
      show: 'libraryDependencies += "com.integrallis" % "vectors" % "0.1.0"',
    },
  };
  // Build-system tabs are the .install-tab buttons that carry a data-target.
  const tabs = document.querySelectorAll('.install-tab[data-target]');
  const cmdText = document.getElementById('install-text');
  const cmdPrompt = document.getElementById('install-prompt');
  const copyBtn = document.getElementById('install-copy');
  let current = 'gradle-kts';

  const renderCmd = (key) => {
    const c = COORDS[key];
    if (!c) return;
    current = key;
    if (cmdText) cmdText.textContent = c.show;
    if (cmdPrompt) cmdPrompt.style.display = 'none';
  };
  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      tabs.forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      renderCmd(tab.dataset.target);
    });
  });
  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(COORDS[current].copy).then(() => {
        copyBtn.classList.add('copied');
        setTimeout(() => copyBtn.classList.remove('copied'), 1500);
      });
    });
  }
});
