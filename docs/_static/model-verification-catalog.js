// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

document.addEventListener("DOMContentLoaded", () => {
  const filters = document.querySelector("#verification-catalog-filters");
  if (!filters) return;

  const table = filters.parentElement.querySelector("table");
  if (!table) return;

  const rows = Array.from(table.querySelectorAll("tbody tr"));
  const search = filters.querySelector("#verification-catalog-search");
  const count = filters.querySelector("#verification-catalog-count");
  const selects = Array.from(filters.querySelectorAll("select[data-catalog-column]"));

  selects.forEach((select) => {
    const column = Number(select.dataset.catalogColumn);
    const values = new Set(rows.map((row) => row.cells[column].textContent.trim()));
    Array.from(values)
      .sort((left, right) => left.localeCompare(right))
      .forEach((value) => select.add(new Option(value, value)));
  });

  const applyFilters = () => {
    const query = search.value.trim().toLocaleLowerCase();
    let visible = 0;
    rows.forEach((row) => {
      const matchesSearch = !query || row.textContent.toLocaleLowerCase().includes(query);
      const matchesSelects = selects.every((select) => {
        const column = Number(select.dataset.catalogColumn);
        return !select.value || row.cells[column].textContent.trim() === select.value;
      });
      row.hidden = !(matchesSearch && matchesSelects);
      if (!row.hidden) visible += 1;
    });
    count.textContent = `${visible} of ${rows.length} concrete configurations`;
  };

  search.addEventListener("input", applyFilters);
  selects.forEach((select) => select.addEventListener("change", applyFilters));
  filters.hidden = false;
  applyFilters();
});
