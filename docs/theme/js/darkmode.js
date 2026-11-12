/*
Copyright © 2014-present, Tom Christie. All rights reserved.

Redistribution and use in source and binary forms, with or
without modification, are permitted provided that the following
conditions are met:

Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/
// MkDocs 1.6.1 theme script with optional-highlight guards and shared SVG theme support.
function setColorMode(mode) {
    // Switch between light/dark theme. `mode` is a string value of either 'dark' or 'light'.
    var hljs_light = document.getElementById('hljs-light'),
        hljs_dark = document.getElementById('hljs-dark');
    document.documentElement.setAttribute('data-bs-theme', mode);
    updateDiagramTheme(mode);
    if (hljs_light && hljs_dark && mode == 'dark') {
        hljs_light.disabled = true;
        hljs_dark.disabled = false;
    } else if (hljs_light && hljs_dark) {
        hljs_dark.disabled = true;
        hljs_light.disabled = false;
    }
}

function updateModeToggle(mode) {
    // Update icon and toggle checkmarks of color mode selector.
    var menu = document.getElementById('theme-menu');
    document.querySelectorAll('[data-bs-theme-value]')
       .forEach(function(toggle) {
            if (mode == toggle.getAttribute('data-bs-theme-value')) {
                toggle.setAttribute('aria-pressed', 'true');
                toggle.lastElementChild.classList.remove('d-none');
                menu.firstElementChild.setAttribute('class', toggle.firstElementChild.getAttribute('class'));
            } else {
                toggle.setAttribute('aria-pressed', 'false');
                toggle.lastElementChild.classList.add('d-none');
            }
        });
}

function onSystemColorSchemeChange(event) {
    // Update site color mode to match system color mode.
    setColorMode(event.matches ? 'dark' : 'light');
}

var mql = window.matchMedia('(prefers-color-scheme: dark)'),
    defaultMode = document.documentElement.getAttribute('data-bs-theme'),
    storedMode = localStorage.getItem('mkdocs-colormode');
if (storedMode && storedMode != 'auto') {
    setColorMode(storedMode);
    updateModeToggle(storedMode);
} else if (storedMode == 'auto' || defaultMode == 'auto') {
    setColorMode(mql.matches ? 'dark' : 'light');
    updateModeToggle('auto');
    mql.addEventListener('change', onSystemColorSchemeChange);
} else {
    setColorMode(defaultMode);
    updateModeToggle(defaultMode);
}

document.querySelectorAll('[data-bs-theme-value]')
    .forEach(function(toggle) {
        toggle.addEventListener('click', function (e) {
            var mode = e.currentTarget.getAttribute('data-bs-theme-value');
            localStorage.setItem('mkdocs-colormode', mode);
            if (mode == 'auto') {
                setColorMode(mql.matches ? 'dark' : 'light');
                mql.addEventListener('change', onSystemColorSchemeChange);
            } else {
                setColorMode(mode);
                mql.removeEventListener('change', onSystemColorSchemeChange);
            }
            updateModeToggle(mode);
        });
    });

function updateDiagramTheme(mode) {
    document.querySelectorAll('img[src*="assets/"][src*=".svg"], a[href*="assets/"][href*=".svg"]')
        .forEach(function(element) {
            var attribute = element.tagName === 'IMG' ? 'src' : 'href';
            var path = element.getAttribute(attribute).split('#')[0];
            if (path.endsWith('.svg')) {
                element.setAttribute(attribute, path + (mode === 'dark' ? '#dark' : ''));
            }
        });
}

document.addEventListener('DOMContentLoaded', function () {
    updateDiagramTheme(document.documentElement.getAttribute('data-bs-theme'));
});
