class TaxiMemory:
    def __init__(self) -> None:
        pass

    def reset(self, state: tuple) -> None:
        self.passenger_pos = None
        self.stations_pos = [
            (state[2], state[3]),
            (state[4], state[5]),
            (state[6], state[7]),
            (state[8], state[9]),
        ]
        self.passenger_picked_up = False
        self.destination_mask = [0, 0, 0, 0]
        self.visit_mask = [1, 1, 1, 1]

    def get_state(self, state) -> tuple:
        taxi_pos = (state[0], state[1])
        at_station = taxi_pos in self.stations_pos
        at_goal = at_station and state[14] == 1
        at_destination = at_station and state[15] == 1
        return (at_goal, at_destination, self.passenger_picked_up, *self.destination_mask, *self.visit_mask)

    @staticmethod
    def near(pos_a, pos_b) -> bool:
        distance = abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])
        return distance <= 1

    def update(self, state, action) -> bool:
        taxi_pos = (state[0], state[1])
        passenger_discovered = state[14]
        destination_discovered = state[15]

        if action in (0, 1, 2, 3):  # MOVE
            for i, station_pos in enumerate(self.stations_pos):
                # if self.near(taxi_pos, station_pos):
                #     if destination_discovered:
                #         self.destination_mask[i] = 1
                if taxi_pos == station_pos:
                    self.visit_mask[i] = 0
                    if destination_discovered:
                        self.destination_mask[i] = 1
                    break
        elif action == 4:  # PICKUP
            if self.passenger_picked_up:
                # print(f"Debug: already picked up")
                pass
            elif passenger_discovered:
                if self.passenger_pos != None:
                    if taxi_pos == self.passenger_pos:
                        # print(f"Debug: picked up again at {taxi_pos}")
                        self.passenger_picked_up = True
                elif taxi_pos in self.stations_pos:
                    # print(f"Debug: discover passenger as {taxi_pos}")
                    self.passenger_pos = taxi_pos
                    self.passenger_picked_up = True
        elif action == 5:  # DROPOFF
            if self.passenger_picked_up:
                self.passenger_picked_up = False
                self.passenger_pos = taxi_pos
                # print(f"Debug: drop at {taxi_pos}")
        return self.get_state(state)