// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const setupModelExplorer = (explorer) => {
  const controls = explorer.querySelector(".verification-model-controls");
  const combinationList = explorer.querySelector(".verification-combination-list");
  const tabs = Array.from(explorer.querySelectorAll("[data-capability-tab]"));
  const precisionButtons = Array.from(explorer.querySelectorAll("[data-precision]"));
  const hardwareButtons = Array.from(explorer.querySelectorAll("[data-hardware]"));
  const combinations = Array.from(explorer.querySelectorAll("[data-entry]"));
  const details = Array.from(explorer.querySelectorAll("[data-entry-detail]"));
  const count = explorer.querySelector(".verification-combination-count");
  const hashEntry = window.location.hash.slice(1);
  const hashCombination = combinations.find((combination) => combination.dataset.entry === hashEntry);
  let activeCapability = hashCombination?.dataset.capability || tabs.find((tab) => !tab.disabled)?.dataset.capabilityTab;
  let activePrecision = hashCombination?.dataset.precision || "";
  let activeHardware = hashCombination?.dataset.hardware || "";
  let activeEntry = hashCombination?.dataset.entry || "";

  const selectEntry = (entryId, updateUrl = false) => {
    activeEntry = entryId;
    combinations.forEach((combination) => {
      const selected = combination.dataset.entry === activeEntry;
      combination.classList.toggle("is-selected", selected);
      combination.setAttribute("aria-pressed", String(selected));
    });
    details.forEach((detail) => {
      detail.hidden = detail.dataset.entryDetail !== activeEntry;
    });
    if (updateUrl) history.replaceState(null, "", `#${activeEntry}`);
  };

  const refreshCombinations = () => {
    const capabilityCombinations = combinations.filter(
      (combination) => combination.dataset.capability === activeCapability,
    );
    const availablePrecisions = new Set(
      capabilityCombinations.map((combination) => combination.dataset.precision).filter(Boolean),
    );
    const availableHardware = new Set(
      capabilityCombinations.map((combination) => combination.dataset.hardware).filter(Boolean),
    );
    if (activePrecision && !availablePrecisions.has(activePrecision)) activePrecision = "";
    if (activeHardware && !availableHardware.has(activeHardware)) activeHardware = "";

    precisionButtons.forEach((button) => {
      const precision = button.dataset.precision;
      button.disabled = Boolean(precision && !availablePrecisions.has(precision));
      button.classList.toggle("is-active", precision === activePrecision);
      button.setAttribute("aria-pressed", String(precision === activePrecision));
    });
    hardwareButtons.forEach((button) => {
      const hardware = button.dataset.hardware;
      button.disabled = Boolean(hardware && !availableHardware.has(hardware));
      button.classList.toggle("is-active", hardware === activeHardware);
      button.setAttribute("aria-pressed", String(hardware === activeHardware));
    });

    const visibleCombinations = [];
    combinations.forEach((combination) => {
      const matchesCapability = combination.dataset.capability === activeCapability;
      const matchesPrecision = !activePrecision || combination.dataset.precision === activePrecision;
      const matchesHardware = !activeHardware || combination.dataset.hardware === activeHardware;
      combination.hidden = !(matchesCapability && matchesPrecision && matchesHardware);
      if (!combination.hidden) visibleCombinations.push(combination);
    });
    count.textContent = `${visibleCombinations.length} combination${visibleCombinations.length === 1 ? "" : "s"}`;
    if (!visibleCombinations.some((combination) => combination.dataset.entry === activeEntry)) {
      activeEntry =
        visibleCombinations.find((combination) => combination.dataset.status === "verified")?.dataset.entry ||
        visibleCombinations[0]?.dataset.entry ||
        "";
    }
    selectEntry(activeEntry);
  };

  const activateCapability = (capability) => {
    activeCapability = capability;
    tabs.forEach((tab) => {
      const selected = tab.dataset.capabilityTab === capability;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    refreshCombinations();
  };

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activateCapability(tab.dataset.capabilityTab));
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const enabledTabs = tabs.filter((candidate) => !candidate.disabled);
      const enabledIndex = enabledTabs.indexOf(tabs[index]);
      const offset = event.key === "ArrowRight" ? 1 : -1;
      const nextTab = enabledTabs[(enabledIndex + offset + enabledTabs.length) % enabledTabs.length];
      nextTab.focus();
      activateCapability(nextTab.dataset.capabilityTab);
    });
  });
  precisionButtons.forEach((button) => {
    button.addEventListener("click", () => {
      activePrecision = button.dataset.precision;
      refreshCombinations();
    });
  });
  hardwareButtons.forEach((button) => {
    button.addEventListener("click", () => {
      activeHardware = button.dataset.hardware;
      refreshCombinations();
    });
  });
  combinations.forEach((combination) => {
    combination.addEventListener("click", () => selectEntry(combination.dataset.entry, true));
  });
  explorer.querySelectorAll(".verification-copy-command").forEach((button) => {
    button.addEventListener("click", async () => {
      const command = button.closest(".verification-command").querySelector("code").textContent;
      await navigator.clipboard.writeText(command);
      button.textContent = "Copied";
      window.setTimeout(() => {
        button.textContent = "Copy";
      }, 1600);
    });
  });

  explorer.classList.add("is-enhanced");
  controls.hidden = false;
  combinationList.hidden = false;
  activateCapability(activeCapability);
};

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-model-explorer]").forEach(setupModelExplorer);
});
