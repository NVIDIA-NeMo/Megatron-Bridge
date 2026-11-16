/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

import io.ray.shaded.com.fasterxml.jackson.core.JsonProcessingException;
import io.ray.shaded.com.fasterxml.jackson.databind.ObjectMapper;
import javax.xml.datatype.Duration;
import javax.xml.datatype.XMLGregorianCalendar;

/** Exercise Ray's actual relocated Jackson classes without starting a cluster. */
public final class RayJacksonSmokeTest {
    private static void rejectsLongNumber(ObjectMapper mapper, String value, Class<?> type)
            throws Exception {
        try {
            mapper.readValue("\"" + value + "\"", type);
        } catch (JsonProcessingException expected) {
            if (!expected.getMessage().contains("exceeds the maximum allowed")) {
                throw new AssertionError("Unexpected parsing failure", expected);
            }
            return;
        }
        throw new AssertionError("Oversized numeric component was accepted: " + type.getName());
    }

    public static void main(String[] args) throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        if (mapper.readValue("\"P1Y\"", Duration.class).getYears() != 1) {
            throw new AssertionError("Normal duration parsing failed");
        }
        XMLGregorianCalendar calendar = mapper.readValue("\"2026-01-01T00:00:00Z\"", XMLGregorianCalendar.class);
        if (calendar.getYear() != 2026) {
            throw new AssertionError("Normal calendar parsing failed");
        }
        rejectsLongNumber(mapper, "P" + "9".repeat(1001) + "Y", Duration.class);
        rejectsLongNumber(mapper, "2026-01-01T00:00:00." + "9".repeat(1001) + "Z", XMLGregorianCalendar.class);
        Class.forName("io.ray.api.Ray", false, RayJacksonSmokeTest.class.getClassLoader());
    }
}
