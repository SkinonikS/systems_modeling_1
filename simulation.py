from __future__ import annotations
from typing import NamedTuple
import scipy.stats as st

class EventStatsType(NamedTuple):
    collected: EventStatsItem
    calculated: EventCalculatedStatsItem

class SimulationStats:
    def __init__(
        self,
        total_completed_event_count: int = 0,
        total_left_in_queue: int = 0,
        total_immediate_service_count: int = 0,
        total_arrived_event_count: int = 0,
        server_idle_item: float = 0.0,
        server_idle_coefficient: float = 0.0,
        average_waiting_time: float = 0.0,
        average_service_time: float = 0.0,
        probability_immediate_service: float = 0.0,
        event_stats: dict[str, EventStatsType] = {},
    ) -> None:
        self.total_completed_event_count = total_completed_event_count
        self.total_left_in_queue = total_left_in_queue
        self.total_immediate_service_count = total_immediate_service_count
        self.total_arrived_event_count = total_arrived_event_count
        self.server_idle_item = server_idle_item
        self.server_idle_coefficient = server_idle_coefficient
        self.average_waiting_time = average_waiting_time
        self.average_service_time = average_service_time
        self.probability_immediate_service = probability_immediate_service
        self.event_stats = event_stats

class EventCalculatedStatsItem:
    def __init__(
        self,
        average_waiting_time: float,
        average_service_time: float,
        left_in_queue: int,
    ) -> None:
        self.left_in_queue = left_in_queue
        self.average_waiting_time = average_waiting_time
        self.average_service_time = average_service_time

class EventStatsItem:
    def __init__(
        self,
        waiting_time: float = 0.0,
        service_time: float = 0.0,
        immediate_service_count: int = 0,
        arrived_event_count: int = 0,
        max_event_queue_size: int = 0,
        completed_event_count: int = 0,
    ) -> None:
        self.waiting_time = waiting_time
        self.service_time = service_time
        self.immediate_service_count = immediate_service_count
        self.arrived_event_count = arrived_event_count
        self.max_event_queue_size = max_event_queue_size
        self.completed_event_count = completed_event_count

class SimulationResult:
    def __init__(
        self,
        event_log: EventLog,
        simulation_stats: SimulationStats,
    ) -> None:
        self.event_log = event_log
        self.simulation_stats = simulation_stats

class EventLogItem:
    def __init__(
        self,
        event_type: str,
        current_time: float = 0.0,
        end_time: float = 0.0,
        is_server_busy: float = False,
        event_log: EventLog = [],
        event_queue: EventQueue = {},
        event_arrival_times: EventArrivalTimes = {},
    ) -> None:
        self.event_type = event_type
        self.is_server_busy = is_server_busy
        self.current_time = current_time
        self.end_time = end_time
        self.event_log = event_log.copy()
        self.event_queue = event_queue.copy()
        self.event_arrival_times = event_arrival_times.copy()

class SimulationData:
    def __init__(
        self,
        event_log: EventLog,
        event_queue: EventQueue,
        event_stats: dict[str, EventStatsItem],
        event_arrival_times: dict[str, float],
        is_server_busy: bool = False,
        server_idle_time: float = 0.0,
        current_time: float  = 0.0,
        end_time: float = 0.0,
    ) -> None:
        self.event_log = event_log
        self.event_stats = event_stats
        self.event_queue = event_queue
        self.event_arrival_times = event_arrival_times
        self.server_idle_time = server_idle_time
        self.current_time = current_time
        self.end_time = end_time
        self.is_server_busy = is_server_busy

class ArrivalEventHandler:
    def __init__(
        self,
        event_type: str,
        arrival_time_generator: st.rv_frozen,
        service_time_generator: st.rv_frozen,
    ) -> None:
        self._event_type = event_type
        self._arrival_time_generator = arrival_time_generator
        self._service_time_generator = service_time_generator

    def handle(self, simulation_time: float, simulation_data: SimulationData) -> None:
        event_stats = simulation_data.event_stats[self._event_type]

        if simulation_data.is_server_busy == False:
            simulation_data.is_server_busy = True

            service_time = self._service_time_generator.rvs()
            simulation_data.end_time = simulation_data.current_time + service_time

            event_stats.completed_event_count += 1
            event_stats.immediate_service_count += 1
            event_stats.service_time += service_time
        else:
            simulation_data.event_queue.append((self._event_type, simulation_data.current_time))
        
        event_stats.arrived_event_count += 1
        event_stats.max_event_queue_size = max(
            event_stats.max_event_queue_size,
            sum(1 for x in simulation_data.event_queue if x[0] == self._event_type),
        )

        simulation_data.event_arrival_times[self._event_type] = simulation_data.current_time + self._arrival_time_generator.rvs()

        simulation_data.event_log.append(EventLogItem(
            event_type=self._event_type,
            current_time=simulation_data.current_time,
            end_time=simulation_data.end_time,
            is_server_busy=simulation_data.is_server_busy,
            event_queue=simulation_data.event_queue,
            event_arrival_times=simulation_data.event_arrival_times,
            event_log=simulation_data.event_log,
        ))

    def get_service_time_generator(self) -> st.rv_frozen:
        return self._service_time_generator
    
    def get_arrival_time_generator(self) -> st.rv_frozen:
        return self._arrival_time_generator

    def get_event_type(self) -> str:
        return self._event_type

class DepartureEventHandler:
    def __init__(
        self,
        event_type: str,
    ) -> None:
        self._event_type = event_type

    def handle(
        self,
        simulation_instance: Simulation,
        simulation_time: float,
        simulation_data: SimulationData,
    ) -> None:
        if len(simulation_data.event_queue) > 0:
            queue_event_type, queue_event_arrival_time = simulation_data.event_queue.pop()
            arrival_event_handler = simulation_instance.get_arrival_event_handler(queue_event_type)

            if arrival_event_handler != None:
                queue_event_wait_time = simulation_data.current_time - queue_event_arrival_time
                event_stats = simulation_data.event_stats[queue_event_type]
                
                service_time = arrival_event_handler.get_service_time_generator().rvs()

                simulation_data.end_time = simulation_data.current_time + service_time

                event_stats.completed_event_count += 1
                event_stats.waiting_time += queue_event_wait_time
                event_stats.service_time += queue_event_wait_time + service_time
        else:
            simulation_data.is_server_busy = False
            simulation_data.end_time = simulation_time + 1

        simulation_data.event_log.append(EventLogItem(
            event_type=self._event_type,
            current_time=simulation_data.current_time,
            end_time=simulation_data.end_time,
            is_server_busy=simulation_data.is_server_busy,
            event_queue=simulation_data.event_queue,
            event_arrival_times=simulation_data.event_arrival_times,
            event_log=simulation_data.event_log,
        ))

    def get_event_type(self) -> str:
        return self._event_type

class Simulation:
    _arrival_event_handlers: dict[str, ArrivalEventHandler] = {}

    def __init__(
        self,
        departure_event_handler: DepartureEventHandler,
        arrival_event_handlers: list[ArrivalEventHandler],
    ) -> None:
        self._departure_event_handler = departure_event_handler

        for e in arrival_event_handlers:
            self.add_arrival_event_handler(e)

    def run_simulation(self, simulation_time: float) -> SimulationResult:
        simulation_data = SimulationData(
            end_time=(simulation_time + 1),
            event_stats=EventStats(),
            event_log=EventLog(),
            event_queue=EventQueue(),
            event_arrival_times=EventArrivalTimes(),
        )

        for key, e in self._arrival_event_handlers.items():
            simulation_data.event_stats[key] = EventStatsItem()
            simulation_data.event_arrival_times[key] = e.get_arrival_time_generator().rvs()

        while simulation_data.current_time < simulation_time:
            next_event_list = (simulation_data.event_arrival_times | {None: simulation_data.end_time}).items()
            next_event_type, next_event_arrival_time = min(next_event_list, key=lambda x: x[1])

            if simulation_data.is_server_busy == False:
                simulation_data.server_idle_time += next_event_arrival_time - simulation_data.current_time
            
            simulation_data.current_time = next_event_arrival_time

            arrival_event_handler = self.get_arrival_event_handler(next_event_type)
            
            if arrival_event_handler != None:
                arrival_event_handler.handle(
                    simulation_time=simulation_time,
                    simulation_data=simulation_data,
                )
            else:
                self._departure_event_handler.handle(
                    simulation_instance=self,
                    simulation_time=simulation_time,
                    simulation_data=simulation_data,
                )

            if simulation_data.current_time >= simulation_time:
                break
        
        simulation_data.current_time = simulation_time

        return SimulationResult(
            event_log=simulation_data.event_log,
            simulation_stats=self._calculate_simulation_stats(simulation_data)
        )

    def get_registered_arrival_event_handlers(self) -> list[str]:
        return list(self._arrival_event_handlers.keys())

    def get_arrival_event_handler(self, event_type: str) -> ArrivalEventHandler | None:
        return self._arrival_event_handlers.get(event_type)

    def add_arrival_event_handler(self, arrival_event_handler: ArrivalEventHandler) -> None:
        self._arrival_event_handlers[arrival_event_handler.get_event_type()] = arrival_event_handler

    def _calculate_simulation_stats(self, simulation_data: SimulationData) -> SimulationStats:
        simulation_stats = SimulationStats()
        
        total_service_time = 0.0
        total_waiting_time = 0.0
        total_arrived_event_count = 0
        total_immediate_service_count = 0
        total_left_in_queue = 0
        total_completed_event_count = 0

        for key in self._arrival_event_handlers.keys():
            event_stats = simulation_data.event_stats.get(key)

            if event_stats == None:
                continue
            
            left_in_queue = sum(1 for x in simulation_data.event_queue if x[0] == key)

            total_left_in_queue += left_in_queue
            total_immediate_service_count += event_stats.immediate_service_count
            total_service_time += event_stats.service_time
            total_waiting_time += event_stats.waiting_time
            total_arrived_event_count += event_stats.arrived_event_count
            total_completed_event_count += event_stats.completed_event_count
            
            calcualted_event_stats = EventCalculatedStatsItem(
                left_in_queue=left_in_queue,
                average_service_time=(event_stats.service_time / event_stats.arrived_event_count) if event_stats.arrived_event_count > 0 else 0,
                average_waiting_time=(event_stats.waiting_time / event_stats.arrived_event_count) if event_stats.arrived_event_count > 0 else 0
            )

            simulation_stats.event_stats[key] = EventStatsType(
                collected=event_stats,
                calculated=calcualted_event_stats
            )
        
        simulation_stats.total_left_in_queue = total_left_in_queue
        simulation_stats.total_completed_event_count = total_completed_event_count
        simulation_stats.total_immediate_service_count = total_immediate_service_count
        simulation_stats.total_arrived_event_count = total_arrived_event_count
        simulation_stats.probability_immediate_service = total_immediate_service_count / total_arrived_event_count
        simulation_stats.average_waiting_time = total_waiting_time / total_arrived_event_count
        simulation_stats.average_service_time = total_service_time / total_arrived_event_count
        simulation_stats.server_idle_item = simulation_data.server_idle_time
        simulation_stats.server_idle_coefficient = simulation_data.server_idle_time / simulation_data.current_time

        return simulation_stats  

class EventStats(dict[str, EventStatsItem]):
    pass

class EventArrivalTimes(dict[str, float]):
    pass

class EventQueue(list[tuple[str, float]]):
    pass

class EventLog(list[EventLogItem]):
    pass

__all__ = [
    'Simulation'
    'ArrivalEventHandler',
    'DepartureEventHandler',
]