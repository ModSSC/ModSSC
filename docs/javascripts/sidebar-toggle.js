(() => {
  const mediaQuery = window.matchMedia("(max-width: 76.1875em)");

  const getDrawer = () => document.getElementById("__drawer");

  const applyDesktopVisible = (visible) => {
    if (mediaQuery.matches) {
      return;
    }
    document.body.classList.toggle("sidebar-visible", visible);
  };

  const applyMobileVisible = (visible) => {
    if (!mediaQuery.matches) {
      return;
    }
    const drawer = getDrawer();
    if (drawer) {
      drawer.checked = Boolean(visible);
    }
  };

  const currentVisible = () => {
    if (mediaQuery.matches) {
      const drawer = getDrawer();
      return drawer ? drawer.checked : false;
    }
    return document.body.classList.contains("sidebar-visible");
  };

  const updateButton = (button) => {
    const visible = currentVisible();
    button.classList.toggle("is-open", visible);
    button.setAttribute("aria-pressed", String(visible));
    button.setAttribute("aria-label", visible ? "Hide sidebar" : "Show sidebar");

    const label = button.querySelector(".sidebar-toggle__text");
    if (label) {
      label.textContent = visible ? "Close" : "Menu";
    }
  };

  const setDefaultState = () => {
    applyDesktopVisible(true);
    applyMobileVisible(false);
  };

  const init = () => {
    if (document.querySelector(".sidebar-toggle")) {
      return;
    }

    const button = document.createElement("button");
    button.type = "button";
    button.className = "sidebar-toggle";
    button.innerHTML =
      '<span class="sidebar-toggle__icon" aria-hidden="true"></span>' +
      '<span class="sidebar-toggle__text">Menu</span>';

    setDefaultState();
    updateButton(button);

    button.addEventListener("click", () => {
      const nextVisible = !currentVisible();
      if (mediaQuery.matches) {
        applyMobileVisible(nextVisible);
      } else {
        applyDesktopVisible(nextVisible);
      }
      updateButton(button);
    });

    const drawer = getDrawer();
    if (drawer) {
      drawer.addEventListener("change", () => updateButton(button));
    }

    mediaQuery.addEventListener("change", () => {
      updateButton(button);
    });

    const header = document.querySelector(".md-header__inner");
    if (header) {
      header.insertBefore(button, header.firstChild);
    } else {
      document.body.appendChild(button);
    }

    const materialDocs = window.document$;
    if (materialDocs && typeof materialDocs.subscribe === "function") {
      materialDocs.subscribe(() => {
        updateButton(button);
      });
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
