from app.context.business_rules import (
    BusinessRule,
    load_business_rules,
    match_business_rules,
    render_business_rules,
    required_tables,
)


def test_load_business_rules_reads_the_six_global_rules() -> None:
    rules = load_business_rules()

    assert {rule.id for rule in rules} == {
        "geografia_estado",
        "ordenes_entregadas",
        "agrupacion_mensual",
        "distribucion_resenas_por_calificacion",
        "categorias_producto",
        "compradores_unicos",
    }


def test_match_business_rules_activates_on_plain_term() -> None:
    matched = match_business_rules(
        "¿Cuáles son los 5 estados con más órdenes entregadas?"
    )

    matched_ids = {rule.id for rule in matched}

    assert "geografia_estado" in matched_ids
    assert "ordenes_entregadas" in matched_ids


def test_match_business_rules_ignores_accents_and_case() -> None:
    matched = match_business_rules("¿CUÁNTOS COMPRADORES RECURRENTES hay?")

    assert any(rule.id == "compradores_unicos" for rule in matched)


def test_match_business_rules_respects_word_boundaries() -> None:
    # "estadounidense" no debe activar la regla de "estado".
    matched = match_business_rules("¿Hay algún proveedor estadounidense?")

    assert not any(rule.id == "geografia_estado" for rule in matched)


def test_match_business_rules_respects_exclusion_terms() -> None:
    # golden_018: pregunta por categoría de tickets de soporte, no de
    # producto — no debe arrastrar la tabla de traducción de categorías.
    matched = match_business_rules(
        "¿Cuántos tickets hay por categoría y qué porcentaje fue resuelto?"
    )

    assert not any(rule.id == "categorias_producto" for rule in matched)


def test_monthly_grouping_rule_avoids_sqlite_only_function() -> None:
    matched = match_business_rules(
        "¿Cuántas órdenes entregadas hubo por mes durante 2018?"
    )
    rule = next(rule for rule in matched if rule.id == "agrupacion_mensual")

    assert "DATE_TRUNC" in rule.regla
    assert "strftime" in rule.regla


def test_review_distribution_rule_separates_dimension_and_count_alias() -> None:
    matched = match_business_rules(
        "¿Cuántas reseñas hay para cada calificación del 1 al 5?"
    )
    rule = next(
        rule
        for rule in matched
        if rule.id == "distribucion_resenas_por_calificacion"
    )
    tables = required_tables(
        matched,
        "¿Cuántas reseñas hay para cada calificación del 1 al 5?",
    )

    assert "COUNT(*) AS reviews" in rule.regla
    assert "COUNT(*) AS review_score" in rule.regla
    assert "olist_order_reviews_dataset" in tables


def test_required_tables_uses_default_when_no_alternative_matches() -> None:
    matched = match_business_rules("¿Cuáles son los 5 estados con más órdenes?")
    tables = required_tables(matched, "¿Cuáles son los 5 estados con más órdenes?")

    assert "olist_customers_dataset" in tables


def test_required_tables_uses_alternative_pattern_when_present() -> None:
    question = (
        "¿Cuáles son los 5 estados de vendedores con más registros "
        "de inventario bajo el punto de reorden?"
    )
    matched = match_business_rules(question)
    tables = required_tables(matched, question)

    assert "olist_sellers_dataset" in tables
    assert "olist_customers_dataset" not in tables


def test_required_tables_merges_multiple_matched_rules_without_duplicates() -> None:
    question = "¿Cuáles son las 5 categorías con más ítems vendidos en órdenes entregadas?"
    matched = match_business_rules(question)
    tables = required_tables(matched, question)

    assert tables.count("olist_orders_dataset") == 1
    assert "olist_products_dataset" in tables
    assert "product_category_name_translation" in tables


def test_render_business_rules_returns_empty_string_when_no_rule_matched() -> None:
    assert render_business_rules([]) == ""


def test_render_business_rules_formats_one_bullet_per_rule() -> None:
    rule = BusinessRule(
        id="regla_test",
        activacion=("termino",),
        excluir=(),
        tablas_requeridas=("tabla_x",),
        tablas_alternativas=(),
        regla="Texto de la regla.",
    )

    rendered = render_business_rules([rule])

    assert rendered == "- Texto de la regla."
