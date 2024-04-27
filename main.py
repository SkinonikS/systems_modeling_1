from __future__ import annotations
import scipy.stats as st
import plotly.graph_objects as pl
import typing as t
import streamlit as s
import collections
import simulation
import random
import test_generators

# Constants
K_ERLANG = 3 # l для Erlanga (L1)
LAMBDA_ERLANG = 0.25 # λ для Erlanga (L1)
MU_NORMAL = 20 # µ для нормального распределения (L1)
SIGMA_NORMAL = 1.5 # σ для нормального распределения (L1)
LAMBDA_EXPON = 1.5 # λ для экспоненциального распределения (L2)
MEAN_POISSON = 0.5 # Среднее для Пуассоновского потока (L2)
SIMULATION_TIME = 500.0 # Время моделирования
LAGS = 15 # Кол-во лагов
BINS = 11 # Кол-во интервалов (бинов)
SAMPLE_SIZE = 1000 # Размер генерируемых значений
ALPHA_SIGNIFICANCE_LEVEL = 0.05 # Уровень значимости α

# Generators
def get_l2_service_time_generator()  -> st.rv_frozen:
    return st.expon(scale=1/s.session_state.lambda_expon)

def get_l2_arrival_time_generator() -> st.rv_frozen:
    return st.expon(scale=s.session_state.mean_poisson)

def get_l1_service_time_generator() -> st.rv_frozen:
    return st.norm(loc=s.session_state.mu_normal, scale=s.session_state.sigma_normal)

def get_l1_arrival_time_generator() -> st.rv_frozen:
    return st.erlang(a=s.session_state.k_erlang, scale=1/s.session_state.lambda_erlang)

# Simulation result draw logic
def __draw_simulation_result_plot(simulation_result: simulation.SimulationResult, registered_events: list[str]) -> None:
    current_times: list[float] = []
    total_queue_sizes: list[int] = []
    event_queue_sizes: collections.defaultdict[str, list[int]] = collections.defaultdict(list)

    for i in simulation_result.event_log:
        current_times.append(i.current_time)
        total_queue_sizes.append(len(i.event_queue))

        for e in registered_events:
            event_queue_sizes[e].append(sum(1 for x in i.event_queue if x[0] == e))

    def __generate_color() -> str:
        return 'rgb({},{},{})'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    fig = pl.Figure()

    for key, y in event_queue_sizes.items():
        fig.add_trace(pl.Scatter(
            x=current_times,
            y=y,
            mode='lines+markers',
            name=f'Заявки ({key})',
            line=dict(color=__generate_color()),
            hovertemplate='Время: %{x}<br>Кол-во заявок: %{y}'
        ))

    fig.add_trace(pl.Scatter(
        x=current_times,
        y=total_queue_sizes,
        mode='lines+markers',
        name='Заявки (Всего)',
        line=dict(color='green'),
        hovertemplate='Время: %{x}<br>Кол-во заявок: %{y}'
    ))

    fig.update_layout(
        xaxis=dict(title='Время моделирования', showgrid=True),
        yaxis=dict(title='Количество заявок', showgrid=True),
        legend_title='Тип заявки',
    )

    s.subheader('Динамика количества заявок в очереди')
    s.plotly_chart(fig, use_container_width=True)

def __draw_simulation_result_event_log(simulation_result: simulation.SimulationResult) -> None:
    event_log = []
    event_log_df_column_config = {}
    event_log_df_column_config['tm'] = s.column_config.NumberColumn(label='tm', format='%.4f')
    event_log_df_column_config['h'] = s.column_config.NumberColumn(label='h', format='%.4f')

    for i in simulation_result.event_log:
        event_log_item = {}
        event_log_item['e'] = i.event_type
        event_log_item['tm'] = i.current_time

        for key, l in i.event_arrival_times.items():
            event_log_item[key] = l
            event_log_df_column_config[key] = s.column_config.NumberColumn(label=key, format='%.4f')

        event_log_item['S'] = 'Busy' if i.is_server_busy == True else 'Idle'
        event_log_item['h'] = i.end_time
        event_log_item['n'] = len(i.event_queue)
        event_log_item['Q'] = ','.join(i[0] for i in i.event_queue)

        event_log.append(event_log_item)

    s.subheader('Результаты симуляции')
    s.dataframe(event_log, use_container_width=True, hide_index=False, column_config=event_log_df_column_config)

def __draw_simulation_result_stats(simulation_result: simulation.SimulationResult) -> None:
    simulation_stats_df = [
        ('Всего обработано', simulation_result.simulation_stats.total_completed_event_count),
        ('Всего вхождений', simulation_result.simulation_stats.total_arrived_event_count),
        ('Всего вхождений без очереди', simulation_result.simulation_stats.total_immediate_service_count),
        ('Всего вхождений в очередь', simulation_result.simulation_stats.total_arrived_event_count - simulation_result.simulation_stats.total_immediate_service_count),
        ('Всего осталось в очереди', simulation_result.simulation_stats.total_left_in_queue),
        ('Суммарное время ожидания', simulation_result.simulation_stats.total_waiting_time),
        ('Суммарное время обслуживания', simulation_result.simulation_stats.total_service_time),
        ('Суммарное время простоя сервера', simulation_result.simulation_stats.server_idle_item),
        ('Коэффициэнт простоя сервера', simulation_result.simulation_stats.server_idle_coefficient),
        ('Вероятность обслуживания заявки без очереди', simulation_result.simulation_stats.probability_immediate_service),
    ]

    for key, e in simulation_result.simulation_stats.event_stats.items():
        simulation_stats_df.append((f'Обработано ({key})', e.collected.completed_event_count))
        simulation_stats_df.append((f'Вхождений ({key})', e.collected.arrived_event_count))
        simulation_stats_df.append((f'Вхождений в очередь ({key})', e.collected.arrived_event_count - e.collected.immediate_service_count))
        simulation_stats_df.append((f'Вхождений без очереди ({key})', e.collected.immediate_service_count))
        simulation_stats_df.append((f'Осталось в очереди ({key})', e.calculated.left_in_queue))
        simulation_stats_df.append((f'Максимальное количество в очереди ({key})', e.collected.max_event_queue_size))
        simulation_stats_df.append((f'Среднее время обработки ({key})', e.calculated.average_service_time))
        simulation_stats_df.append((f'Среднее время ожидания в очереди ({key})', e.calculated.average_waiting_time))

    s.subheader('Статистика симуляции')
    s.dataframe(simulation_stats_df, use_container_width=True, column_config={
        1: s.column_config.TextColumn(label='Название'),
        2: s.column_config.NumberColumn(label='Значение', format='%.4f'),
    })

# Generators test draw logic
class TestsResults(t.NamedTuple):
    dist: test_generators.DistributionTestResult
    comp: test_generators.CompareParametersResult
    acf: test_generators.AutocorrelationFunctionResult
    ks_chi: test_generators.KolmagorovSmirnovChiSquareTestResult

def __draw_generators_test_distribution_plot(test_results: test_generators.TesterResult[TestsResults], test_config: test_generators.TestConfig) -> None:
    fig = pl.Figure()

    fig.add_trace(pl.Bar(
        x=test_results.results.dist.bin_edges,
        y=test_results.results.dist.hist,
        name='Эмпирическое',
        marker_color='green',
        hovertemplate='Частота: %{y}<br>Значение: %{x}',
    ))

    fig.add_trace(pl.Scatter(
        x=test_results.results.dist.x,
        y=test_results.results.dist.y,
        mode='lines',
        name='Теоретическое',
        line=dict(color='lightskyblue', width=2),
        hovertemplate='Частота: %{y}<br>Значение: %{x}'
    ))

    fig.update_layout(
        xaxis_title='Значение',
        yaxis_title='Частота',
        legend_title='Распределение',
        bargap=0.01
    )

    s.plotly_chart(fig, use_container_width=True)

def __draw_generators_test_compare_parameters_df(test_results: test_generators.TesterResult[TestsResults]) -> None:
    s.subheader('Сравнение параметров распределения')
    s.dataframe((
        ('Значение мю (μ)', test_results.results.comp.expected_mu, test_results.results.comp.actual_mu),
        ('Значение сигма (σ)', test_results.results.comp.expected_sigma, test_results.results.comp.actual_sigma),
    ), use_container_width=True, column_config={
        1: s.column_config.TextColumn(label='Название'),
        2: s.column_config.NumberColumn(label='Ожидаемое значение', format='%.04f'),
        3: s.column_config.NumberColumn(label='Полученное значение', format='%.04f'),
    })

def __draw_generators_test_acf_plot(test_results: test_generators.TesterResult[TestsResults]) -> None:
    fig = pl.Figure()

    fig.add_trace(pl.Bar(
        x=test_results.results.acf.lags,
        y=test_results.results.acf.acf_values,
        name='Автокорреляция',
        marker_color='lightblue',
        showlegend=False,
        hovertemplate='Лаг: %{x}<br>Значение: %{y}'
    ))

    fig.add_trace(pl.Scatter(
        x=test_results.results.acf.lags,
        y=test_results.results.acf.upper_confidence,
        mode='lines',
        name='Верхний предел (95%)',
        line=dict(color='red', dash='dash'),
        showlegend=False,
        hovertemplate='Лаг: %{x}<br>Значение: %{y}'
    ))

    fig.add_trace(pl.Scatter(
        x=test_results.results.acf.lags,
        y=test_results.results.acf.lower_confidence,
        mode='lines',
        name='Нижний предел (95%)',
        line=dict(color='red', dash='dash'),
        showlegend=False,
        hovertemplate='Лаг: %{x}<br>Значение: %{y}'
    ))

    fig.update_layout(
        xaxis_title='Лаг',
        yaxis_title='Автокорреляция',
        bargap=0.01,
    )

    s.subheader('Проверка независимости значений сгенерированной выборки')
    s.plotly_chart(fig, use_container_width=True)

def __draw_generators_test_ks_chi_square_df(test_results: test_generators.TesterResult[TestsResults], test_config: test_generators.TestConfig) -> None:
    ks, chi_square = test_results.results.ks_chi
    
    s.subheader('Проверка гипотезы о виде закона распределения выборки')
    s.dataframe([
        ('Критерий X2', test_config.alpha, chi_square.criteria, chi_square.df, '(0; {:.4f})'.format(chi_square.critical_value), chi_square.p_value),
        ('Критерий Колмогорова Смирнова', test_config.alpha, ks.criteria, None, '(0; {:.4f})'.format(ks.critical_value), ks.p_value),
    ], use_container_width=True, column_config={
        1: s.column_config.TextColumn(label='Название'),
        2: s.column_config.NumberColumn(label='Уровень значимости α', format='%.04f'),
        3: s.column_config.NumberColumn(label='Значение критерия', format='%.04f'),
        4: s.column_config.NumberColumn(label='df', format='%d'),
        5: s.column_config.TextColumn(label='Критическое значение'),
        6: s.column_config.NumberColumn(label='Значение P', format='%.04f'),
    })

# Streamlit initialize logic
def __initialize_inputs() -> None:
    if 'k_erlang' not in s.session_state:
        s.session_state.k_erlang = K_ERLANG
    if 'lambda_erlang' not in s.session_state:
        s.session_state.lambda_erlang = LAMBDA_ERLANG 

    if 'mu_normal' not in s.session_state:
        s.session_state.mu_normal = MU_NORMAL
    if 'sigma_normal' not in s.session_state:
        s.session_state.sigma_normal = SIGMA_NORMAL 

    if 'mean_poisson' not in s.session_state:
        s.session_state.mean_poisson = MEAN_POISSON 

    if 'lambda_expon' not in s.session_state:
        s.session_state.lambda_expon = LAMBDA_EXPON

    if 'simulation_time' not in s.session_state:
        s.session_state.simulation_time = SIMULATION_TIME

    if 'sample_size' not in s.session_state:
        s.session_state.sample_size = SAMPLE_SIZE

    if 'lags' not in s.session_state:
        s.session_state.lags = LAGS

    if 'bins' not in s.session_state:
        s.session_state.bins = BINS

    if 'alpha_significance_level' not in s.session_state:
        s.session_state.alpha_significance_level = ALPHA_SIGNIFICANCE_LEVEL

def __initialize_page() -> None:
    s.set_page_config('Моделирование систем')

# Main. This is where the magic begins...
def main() -> None:
    __initialize_inputs()
    __initialize_page()

    with s.expander('Настройки генераторов', expanded=True):
        s.subheader('Распределение Эрланга', help='Эта функция будет использована для генерации времени прибытия заявки L1')
        col1, col2 = s.columns(2)
        with col1:
            s.number_input(label='Длина (K)', min_value=1, step=1, format='%d', key='k_erlang')
        with col2:
            s.number_input(label='Лямбда (λ)', min_value=0.01, step=0.01, format='%f', key='lambda_erlang')

        s.subheader('Нормальное распределение', help='Эта функция будет использована для генерации времени обработки заявки L1')
        col1, col2 = s.columns(2)
        with col1:
            s.number_input(label='Мю (μ)', min_value=1, step=1, format='%d', key='mu_normal')
        with col2:
            s.number_input(label='Сигма (σ)', min_value=0.01, step=0.01, format='%f', key='sigma_normal')

        s.subheader('Распределение Пуассона', help='Эта функция будет использована для генерации времени прибытия заявки L2')
        s.number_input(label='Среднее', min_value=0.01, step=0.01, format='%f', key='mean_poisson')

        s.subheader('Экспоненциальное распределение', help='Эта функция будет использована для генерации времени обработки заявки L2')
        s.number_input(label='Лямбда (λ)', min_value=0.01, step=0.01, format='%f', key='lambda_expon')

    tab1, tab2 = s.tabs(['🗂️ Моделирование системы', '📈 Тестирование генераторов'])
    with tab1:
        s.header('Моделирование системы')
        with s.expander('Настройки модели', expanded=True):
            s.number_input(label='Время моделирования (Tf)', min_value=1.0, step=0.5, format='%f', key='simulation_time')
        
        if s.button('Запустить моделирование', use_container_width=True):
            simulation_instance = simulation.Simulation(
                departure_event_handler=simulation.DepartureEventHandler(event_type='h'),
                arrival_event_handlers=[
                    simulation.ArrivalEventHandler(
                        event_type='l1',
                        arrival_time_generator=get_l1_arrival_time_generator(),
                        service_time_generator=get_l1_service_time_generator(),
                    ),
                    simulation.ArrivalEventHandler(
                        event_type='l2',
                        arrival_time_generator=get_l2_arrival_time_generator(),
                        service_time_generator=get_l2_service_time_generator(),
                    ),
                ]
            )

            simulation_result = simulation_instance.run_simulation(simulation_time=s.session_state.simulation_time)

            __draw_simulation_result_event_log(simulation_result=simulation_result)
            __draw_simulation_result_stats(simulation_result=simulation_result)
            __draw_simulation_result_plot(simulation_result=simulation_result, registered_events=simulation_instance.get_registered_arrival_event_handlers())

    with tab2:
        s.header('Тестирование генераторов')
        with s.expander('Настройки тестирования', expanded=True):
            s.number_input(label='Размер выборки', min_value=1, step=1, format='%d', key='sample_size')
            s.number_input(label='Количество периодов', min_value=1, step=1, format='%d', key='lags')
            s.number_input(label='Количество столбцов', min_value=1, step=1, format='%d', key='bins')
            s.number_input(label='Уровень значимости α', min_value=0.0, max_value=1.0, step=0.01, format='%f', key='alpha_significance_level')

        if s.button('Запустить тестирование', use_container_width=True):
            dist_headers = {
                'l1_arrival': 'Распределение Эрланга',
                'l1_service': 'Нормальное распределение',
                'l2_arrival': 'Распределение Пуассона',
                'l2_service': 'Экспоненциальное распределение',
            }

            tester = test_generators.Tester[TestsResults](
                tests={
                    'dist': test_generators.DistributionTest(),
                    'comp': test_generators.CompareParametersTest(),
                    'acf': test_generators.AutocorrelationFunctionTest(),
                    'ks_chi': test_generators.KolmagorovSmirnovChiSquareTest(),
                },
                generators={
                    'l1_arrival': get_l1_arrival_time_generator(),
                    'l1_service': get_l1_service_time_generator(),
                    'l2_arrival': get_l2_arrival_time_generator(),
                    'l2_service': get_l2_service_time_generator(),
                },
                result_factory=lambda x: TestsResults(**x),
            )

            test_config = test_generators.TestConfig(
                sample_size=s.session_state.sample_size,
                lags=s.session_state.lags,
                bins=s.session_state.bins,
                alpha=s.session_state.alpha_significance_level,
            )

            test_results = tester.run_tests(config=test_config)

            for key, i in test_results.items():
                header = dist_headers.get(key)
                
                if header != None:
                    s.header(body=header)

                __draw_generators_test_distribution_plot(test_results=i, test_config=test_config)
                __draw_generators_test_compare_parameters_df(test_results=i)
                __draw_generators_test_ks_chi_square_df(test_results=i, test_config=test_config)
                __draw_generators_test_acf_plot(test_results=i)

                s.divider()
        
if __name__ == '__main__':
    main()